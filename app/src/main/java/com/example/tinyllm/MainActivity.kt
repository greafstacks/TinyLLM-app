package com.example.tinyllm

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.provider.OpenableColumns
import android.widget.Button
import android.widget.EditText
import android.widget.SeekBar
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import kotlin.concurrent.thread

/**
 * No bundled model.bin here at all -- the model is picked from device
 * storage at runtime via the system file picker (Storage Access
 * Framework), read as raw bytes, and handed to TransformerModel.load(),
 * exactly the same call that used to read from assets. Picking a new
 * file at any time swaps the model with no rebuild needed.
 *
 * The picked file's URI is remembered (with a persisted read grant,
 * so it survives app restarts and reboots) and reloaded automatically
 * next time the app opens.
 */
class MainActivity : AppCompatActivity() {

    private lateinit var statusText: TextView
    private lateinit var loadModelButton: Button
    private lateinit var input: EditText
    private lateinit var output: TextView
    private lateinit var sendButton: Button
    private lateinit var lengthLabel: TextView
    private lateinit var lengthSeek: SeekBar

    private var model: TransformerModel? = null

    // Must be registered unconditionally during activity construction (not
    // inside onCreate's body, and never inside a click listener) -- that's
    // an Android platform requirement for registerForActivityResult.
    private val pickModelLauncher = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) handlePickedModel(uri)
    }

    /** Slider position 0..98 -> 20..1000 characters, in steps of 10. */
    private fun charsFor(progress: Int): Int = 20 + progress * 10

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        statusText = findViewById(R.id.statusText)
        loadModelButton = findViewById(R.id.loadModelButton)
        input = findViewById(R.id.inputText)
        output = findViewById(R.id.outputText)
        sendButton = findViewById(R.id.sendButton)
        lengthLabel = findViewById(R.id.lengthLabel)
        lengthSeek = findViewById(R.id.lengthSeek)

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
        statusText.text = getString(R.string.no_model)

        loadModelButton.setOnClickListener {
            pickModelLauncher.launch(arrayOf("*/*"))
        }

        sendButton.setOnClickListener {
            val currentModel = model ?: return@setOnClickListener
            val prompt = input.text.toString()
            val maxChars = charsFor(lengthSeek.progress)

            sendButton.isEnabled = false
            val previousStatus = statusText.text
            statusText.text = getString(R.string.generating)

            thread {
                try {
                    val tokens = currentModel.tokenize(prompt)
                    val continuation = currentModel.generate(tokens, maxChars)
                    runOnUiThread {
                        output.text = prompt + continuation
                        statusText.text = previousStatus
                        sendButton.isEnabled = true
                    }
                } catch (e: Throwable) {
                    runOnUiThread {
                        statusText.text = "Generation failed: $e"
                        sendButton.isEnabled = true
                    }
                }
            }
        }

        val savedUri = getPreferences(MODE_PRIVATE).getString(KEY_MODEL_URI, null)
        if (savedUri != null) {
            loadModelFrom(Uri.parse(savedUri))
        }
    }

    private fun handlePickedModel(uri: Uri) {
        try {
            contentResolver.takePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION)
        } catch (e: SecurityException) {
            // Some providers don't support a persistent grant; loading below
            // still works for this session, it just won't auto-reload next launch.
        }
        getPreferences(MODE_PRIVATE).edit().putString(KEY_MODEL_URI, uri.toString()).apply()
        loadModelFrom(uri)
    }

    private fun loadModelFrom(uri: Uri) {
        sendButton.isEnabled = false
        statusText.text = getString(R.string.loading)

        thread {
            try {
                val bytes = contentResolver.openInputStream(uri)?.use { it.readBytes() }
                    ?: throw java.io.IOException("the picker returned no data for that file")
                val loaded = TransformerModel.load(bytes)
                val name = displayNameFor(uri)
                runOnUiThread {
                    model = loaded
                    statusText.text = getString(R.string.model_loaded, name)
                    sendButton.isEnabled = true
                }
            } catch (e: Throwable) {
                runOnUiThread {
                    statusText.text = getString(R.string.load_failed, e.message ?: e.toString())
                    sendButton.isEnabled = false
                }
            }
        }
    }

    private fun displayNameFor(uri: Uri): String {
        var name = uri.lastPathSegment ?: "model"
        contentResolver.query(uri, null, null, null, null)?.use { cursor ->
            val idx = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
            if (idx >= 0 && cursor.moveToFirst()) {
                cursor.getString(idx)?.let { name = it }
            }
        }
        return name
    }

    companion object {
        private const val KEY_MODEL_URI = "model_uri"
    }
}
