package com.example.tinyllm

import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.SeekBar
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import kotlin.concurrent.thread

/**
 * Loads model.bin (trained by training/train.py) on a background thread,
 * then runs the hand-written transformer in Transformer.kt to generate
 * text from whatever the user types. The slider picks how many characters
 * of reply to generate. Generation runs off the main thread, and errors
 * are shown in the output box, since there's no debugger attached when
 * you install an APK straight from GitHub Actions.
 */
class MainActivity : AppCompatActivity() {

    private var model: TransformerModel? = null

    /** Slider position 0..98 -> 20..1000 characters, in steps of 10. */
    private fun charsFor(progress: Int): Int = 20 + progress * 10

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val input = findViewById<EditText>(R.id.inputText)
        val output = findViewById<TextView>(R.id.outputText)
        val sendButton = findViewById<Button>(R.id.sendButton)
        val lengthLabel = findViewById<TextView>(R.id.lengthLabel)
        val lengthSeek = findViewById<SeekBar>(R.id.lengthSeek)

        lengthSeek.max = 98
        lengthSeek.progress = 28 // 300 characters
        lengthLabel.text = getString(R.string.length_label, charsFor(lengthSeek.progress))
        lengthSeek.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                lengthLabel.text = getString(R.string.length_label, charsFor(progress))
            }

            override fun onStartTrackingTouch(seekBar: SeekBar?) {}

            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })

        sendButton.isEnabled = false
        output.text = getString(R.string.loading)

        thread {
            try {
                val bytes = assets.open("model.bin").use { it.readBytes() }
                val loaded = TransformerModel.load(bytes)
                runOnUiThread {
                    model = loaded
                    output.text = getString(R.string.ready)
                    sendButton.isEnabled = true
                }
            } catch (e: Throwable) {
                runOnUiThread { output.text = "Failed to load model: $e" }
            }
        }

        sendButton.setOnClickListener {
            val currentModel = model ?: return@setOnClickListener
            val prompt = input.text.toString()
            val maxChars = charsFor(lengthSeek.progress)

            sendButton.isEnabled = false
            output.text = getString(R.string.generating)

            thread {
                try {
                    val tokens = currentModel.tokenize(prompt)
                    val continuation = currentModel.generate(tokens, maxChars)
                    runOnUiThread {
                        output.text = prompt + continuation
                        sendButton.isEnabled = true
                    }
                } catch (e: Throwable) {
                    runOnUiThread {
                        output.text = "Generation failed: $e"
                        sendButton.isEnabled = true
                    }
                }
            }
        }
    }
}
