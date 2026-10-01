package com.example.tinyllm

import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.exp
import kotlin.math.sqrt
import kotlin.math.tanh
import kotlin.random.Random

/**
 * Inference engine for the tiny GPT-style transformer trained by
 * training/train.py, including its subword (BPE) tokenizer. A direct port
 * of training/model_ref.py (model + file format) and training/bpe.py
 * (tokenizer) -- read those for the reference versions and the tests that
 * check this logic (KV-cache vs. whole-sequence forward pass, window
 * rollover, binary round trip, tokenizer round trip).
 *
 * Weights are flat FloatArrays, row-major, matrices stored (in, out):
 * element [k][o] of a matrix lives at index k * outDim + o.
 *
 * KV-cache: each generated token only computes its own row through the
 * network and attends over the cached keys/values of earlier tokens.
 */

private class Layer(
    val ln1W: FloatArray, val ln1B: FloatArray,
    val qW: FloatArray, val qB: FloatArray,
    val kW: FloatArray, val kB: FloatArray,
    val vW: FloatArray, val vB: FloatArray,
    val projW: FloatArray, val projB: FloatArray,
    val ln2W: FloatArray, val ln2B: FloatArray,
    val fcW: FloatArray, val fcB: FloatArray,
    val fcProjW: FloatArray, val fcProjB: FloatArray,
)

private class Weights(
    val vocabSize: Int, val blockSize: Int, val nEmbd: Int, val nHead: Int, val nLayer: Int,
    val nBase: Int,
    val tokens: Array<String>,          // token id -> text
    val stoiBase: Map<Char, Int>,       // single character -> token id (ids 0 until nBase)
    val ranks: Map<Long, Int>,          // (idA, idB) -> merge number; that merge creates id nBase + number
    val tokEmb: FloatArray, val posEmb: FloatArray,
    val layers: List<Layer>,
    val lnFW: FloatArray, val lnFB: FloatArray,
    val headW: FloatArray, val headB: FloatArray,
)

/** Cached keys/values: layer l, position p, channel d live at keys[l][p * nEmbd + d]. */
private class State(nLayer: Int, blockSize: Int, nEmbd: Int) {
    val keys = Array(nLayer) { FloatArray(blockSize * nEmbd) }
    val values = Array(nLayer) { FloatArray(blockSize * nEmbd) }
    var length = 0
}

private class Reader(bytes: ByteArray) {
    private val buf: ByteBuffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)

    fun magic(): String {
        val b = ByteArray(4)
        buf.get(b)
        return String(b, Charsets.US_ASCII)
    }

    fun int(): Int = buf.getInt()

    fun ushort(): Int = buf.getShort().toInt() and 0xFFFF

    /** uint16 length, then that many uint16 UTF-16 code units. */
    fun string(): String {
        val n = ushort()
        val chars = CharArray(n) { ushort().toChar() }
        return String(chars)
    }

    fun floats(n: Int): FloatArray {
        val out = FloatArray(n)
        buf.asFloatBuffer().get(out) // the view starts at the current position
        buf.position(buf.position() + n * 4)
        return out
    }

    fun remaining(): Int = buf.remaining()
}

private fun pairKey(a: Int, b: Int): Long = (a.toLong() shl 32) or (b.toLong() and 0xFFFFFFFFL)

// ---- math on single vectors (mirrors Session._step in model_ref.py) ----

/** x: (inDim) times w: (inDim, outDim) plus b: (outDim). */
private fun linear(x: FloatArray, w: FloatArray, b: FloatArray, inDim: Int, outDim: Int): FloatArray {
    val out = b.copyOf()
    for (k in 0 until inDim) {
        val xk = x[k]
        val row = k * outDim
        for (o in 0 until outDim) {
            out[o] += xk * w[row + o]
        }
    }
    return out
}

private fun layerNorm(x: FloatArray, w: FloatArray, b: FloatArray): FloatArray {
    val n = x.size
    var mean = 0f
    for (v in x) mean += v
    mean /= n
    var variance = 0f
    for (v in x) {
        val d = v - mean
        variance += d * d
    }
    variance /= n
    val denom = sqrt(variance + 1e-5f)
    return FloatArray(n) { (x[it] - mean) / denom * w[it] + b[it] }
}

private const val GELU_K = 0.7978845608028654f // sqrt(2 / pi)

/** tanh approximation of GELU -- matches nn.GELU(approximate='tanh') in train.py. */
private fun geluInPlace(x: FloatArray) {
    for (i in x.indices) {
        val v = x[i]
        x[i] = 0.5f * v * (1f + tanh(GELU_K * (v + 0.044715f * v * v * v)))
    }
}

class TransformerModel private constructor(private val w: Weights) {

    private val chunkCache = HashMap<String, IntArray>()

    // ---------------- tokenizer (twin of training/bpe.py) ----------------

    /** Merge the lowest-numbered mergeable pair, repeatedly, until none is left. */
    private fun encodeChunk(chunk: String): IntArray {
        chunkCache[chunk]?.let { return it }
        var ids = ArrayList<Int>(chunk.length)
        for (ch in chunk) {
            val id = w.stoiBase[ch]
            if (id != null) ids.add(id) // characters the model never saw are dropped
        }
        while (ids.size > 1) {
            var bestRank = Int.MAX_VALUE
            var bestI = -1
            for (i in 0 until ids.size - 1) {
                val r = w.ranks[pairKey(ids[i], ids[i + 1])]
                if (r != null && r < bestRank) {
                    bestRank = r
                    bestI = i
                }
            }
            if (bestI < 0) break
            val a = ids[bestI]
            val b = ids[bestI + 1]
            val newId = w.nBase + bestRank
            val merged = ArrayList<Int>(ids.size)
            var k = 0
            while (k < ids.size) {
                if (k < ids.size - 1 && ids[k] == a && ids[k + 1] == b) {
                    merged.add(newId)
                    k += 2
                } else {
                    merged.add(ids[k])
                    k += 1
                }
            }
            ids = merged
        }
        val result = ids.toIntArray()
        chunkCache[chunk] = result
        return result
    }

    /**
     * Split into chunks the same way bpe.pretokenize does: a word of letters
     * (with ONE optional leading space), a run of digits, or a single other
     * character -- then BPE-encode each chunk.
     */
    fun tokenize(text: String): List<Int> {
        val out = ArrayList<Int>()
        val n = text.length
        var i = 0
        while (i < n) {
            val ch = text[i]
            var j: Int
            if (ch == ' ' && i + 1 < n && text[i + 1].isLetter()) {
                j = i + 1
                while (j < n && text[j].isLetter()) j++
            } else if (ch.isLetter()) {
                j = i
                while (j < n && text[j].isLetter()) j++
            } else if (ch in '0'..'9') {
                j = i
                while (j < n && text[j] in '0'..'9') j++
            } else {
                j = i + 1
            }
            for (id in encodeChunk(text.substring(i, j))) out.add(id)
            i = j
        }
        return out
    }

    // ---------------- the model ----------------

    /** Process ONE token at the next free cache position; returns logits for the token after it. */
    private fun step(state: State, token: Int): FloatArray {
        val c = w.nEmbd
        val hs = c / w.nHead
        val pos = state.length
        check(pos < w.blockSize) { "KV cache is full" }
        val sqrtHs = sqrt(hs.toFloat())

        val x = FloatArray(c) { w.tokEmb[token * c + it] + w.posEmb[pos * c + it] }

        for (l in 0 until w.nLayer) {
            val layer = w.layers[l]

            val h1 = layerNorm(x, layer.ln1W, layer.ln1B)
            val q = linear(h1, layer.qW, layer.qB, c, c)
            val k = linear(h1, layer.kW, layer.kB, c, c)
            val v = linear(h1, layer.vW, layer.vB, c, c)
            val keys = state.keys[l]
            val values = state.values[l]
            System.arraycopy(k, 0, keys, pos * c, c)
            System.arraycopy(v, 0, values, pos * c, c)

            val att = FloatArray(c)
            for (h in 0 until w.nHead) {
                val lo = h * hs
                val scores = FloatArray(pos + 1)
                var maxScore = Float.NEGATIVE_INFINITY
                for (j in 0..pos) {
                    var dot = 0f
                    val base = j * c + lo
                    for (d in 0 until hs) dot += q[lo + d] * keys[base + d]
                    dot /= sqrtHs
                    scores[j] = dot
                    if (dot > maxScore) maxScore = dot
                }
                var sum = 0f
                for (j in 0..pos) {
                    val e = exp(scores[j] - maxScore)
                    scores[j] = e
                    sum += e
                }
                for (j in 0..pos) {
                    val weight = scores[j] / sum
                    val base = j * c + lo
                    for (d in 0 until hs) att[lo + d] += weight * values[base + d]
                }
            }
            val attOut = linear(att, layer.projW, layer.projB, c, c)
            for (i in 0 until c) x[i] += attOut[i]

            val h2 = layerNorm(x, layer.ln2W, layer.ln2B)
            val hidden = linear(h2, layer.fcW, layer.fcB, c, 4 * c)
            geluInPlace(hidden)
            val mlpOut = linear(hidden, layer.fcProjW, layer.fcProjB, 4 * c, c)
            for (i in 0 until c) x[i] += mlpOut[i]
        }

        val xf = layerNorm(x, w.lnFW, w.lnFB)
        state.length = pos + 1
        return linear(xf, w.headW, w.headB, c, w.vocabSize)
    }

    /**
     * Temperature + top-k sampling. Top-k (only the k most likely tokens are
     * candidates) matters more with a big vocabulary: it stops the rare
     * long-tail tokens from occasionally derailing a sentence.
     */
    private fun sample(logits: FloatArray, temperature: Double, topK: Int, rng: Random): Int {
        val temp = if (temperature < 1e-3) 1e-3 else temperature
        var threshold = Float.NEGATIVE_INFINITY
        if (topK in 1 until logits.size) {
            val sorted = logits.copyOf()
            sorted.sort()
            threshold = sorted[sorted.size - topK]
        }
        var maxLogit = Float.NEGATIVE_INFINITY
        for (v in logits) if (v > maxLogit) maxLogit = v

        val probs = DoubleArray(logits.size)
        var sum = 0.0
        for (i in logits.indices) {
            if (logits[i] < threshold) continue
            val p = exp((logits[i] - maxLogit).toDouble() / temp)
            probs[i] = p
            sum += p
        }
        var r = rng.nextDouble() * sum
        for (i in probs.indices) {
            r -= probs[i]
            if (r <= 0.0 && probs[i] > 0.0) return i
        }
        var best = 0
        for (i in logits.indices) if (logits[i] > logits[best]) best = i
        return best
    }

    /**
     * Generates exactly [maxChars] characters of new text (the caller shows
     * the prompt itself). Tokens are produced one at a time until the text
     * is long enough, then the end is trimmed to the exact length.
     *
     * The model uses absolute position embeddings, so once the cache holds
     * blockSize tokens it can't just slide: it restarts from the most
     * recent half of the context and carries on.
     */
    fun generate(
        promptTokens: List<Int>,
        maxChars: Int,
        temperature: Double = 0.9,
        topK: Int = 40,
        seed: Long? = null,
        stopAt: Char? = null, // chat mode: end the reply at this character (e.g. '\n') instead of running to maxChars
    ): String {
        if (maxChars <= 0) return ""
        val rng = if (seed != null) Random(seed) else Random.Default
        val seedTokens = if (promptTokens.isEmpty()) {
            // Prefer starting from a learned "start of message" boundary (newline) if it exists.
            listOf(w.stoiBase['\n'] ?: rng.nextInt(w.nBase))
        } else {
            promptTokens
        }

        val history = ArrayList<Int>()
        var state = State(w.nLayer, w.blockSize, w.nEmbd)
        var logits = FloatArray(0)

        fun feed(token: Int) {
            if (state.length >= w.blockSize) {
                val keep = w.blockSize / 2
                val tail = history.takeLast(keep)
                state = State(w.nLayer, w.blockSize, w.nEmbd)
                for (t in tail) step(state, t)
            }
            logits = step(state, token)
            history.add(token)
        }

        for (t in seedTokens) feed(t)

        val out = StringBuilder()
        while (out.length < maxChars) {
            val next = sample(logits, temperature, topK, rng)
            out.append(w.tokens[next])
            if (stopAt != null) {
                val stopIdx = out.indexOf(stopAt.toString())
                if (stopIdx >= 0) {
                    out.setLength(stopIdx)
                    return out.toString()
                }
            }
            if (out.length < maxChars) feed(next)
        }
        out.setLength(maxChars)
        return out.toString()
    }

    companion object {
        fun load(bytes: ByteArray): TransformerModel {
            val r = Reader(bytes)
            require(r.magic() == "TLM2") { "model.bin has the wrong header (expected TLM2 -- rebuild the app so the model and code match)" }
            val vocab = r.int()
            val block = r.int()
            val c = r.int()
            val heads = r.int()
            val nLayer = r.int()
            val nBase = r.int()
            require(heads > 0 && c % heads == 0) { "n_embd must be divisible by n_head" }
            require(nBase in 1..vocab) { "bad base vocabulary size" }

            val tokens = Array(vocab) { r.string() }
            val stoiBase = HashMap<Char, Int>()
            for (i in 0 until nBase) stoiBase[tokens[i][0]] = i

            val ranks = HashMap<Long, Int>()
            for (m in 0 until vocab - nBase) {
                val a = r.int()
                val b = r.int()
                ranks[pairKey(a, b)] = m
            }

            val tokEmb = r.floats(vocab * c)
            val posEmb = r.floats(block * c)

            val layers = ArrayList<Layer>(nLayer)
            repeat(nLayer) {
                layers.add(
                    Layer(
                        ln1W = r.floats(c), ln1B = r.floats(c),
                        qW = r.floats(c * c), qB = r.floats(c),
                        kW = r.floats(c * c), kB = r.floats(c),
                        vW = r.floats(c * c), vB = r.floats(c),
                        projW = r.floats(c * c), projB = r.floats(c),
                        ln2W = r.floats(c), ln2B = r.floats(c),
                        fcW = r.floats(c * 4 * c), fcB = r.floats(4 * c),
                        fcProjW = r.floats(4 * c * c), fcProjB = r.floats(c),
                    )
                )
            }

            val lnFW = r.floats(c)
            val lnFB = r.floats(c)
            val headW = r.floats(c * vocab)
            val headB = r.floats(vocab)
            check(r.remaining() == 0) { "model.bin has unexpected trailing bytes" }

            return TransformerModel(
                Weights(
                    vocabSize = vocab, blockSize = block, nEmbd = c, nHead = heads, nLayer = nLayer,
                    nBase = nBase, tokens = tokens, stoiBase = stoiBase, ranks = ranks,
                    tokEmb = tokEmb, posEmb = posEmb, layers = layers,
                    lnFW = lnFW, lnFB = lnFB, headW = headW, headB = headB,
                )
            )
        }
    }
}
