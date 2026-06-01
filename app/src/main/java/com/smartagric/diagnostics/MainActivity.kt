package com.smartagric.diagnostics

import android.content.Intent
import android.os.Bundle
import android.widget.Button
import android.widget.FrameLayout
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import com.smartagric.diagnostics.data.AppDatabase
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : AppCompatActivity() {

    companion object {
        const val EXTRA_CROP = "extra_crop"
        const val CROP_MAIZE = "Maize"
        const val CROP_CASSAVA = "Cassava"
        const val CROP_BEANS = "Beans"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val cardMaize = findViewById<FrameLayout>(R.id.cardMaize)
        val cardCassava = findViewById<FrameLayout>(R.id.cardCassava)
        val cardBeans = findViewById<FrameLayout>(R.id.cardBeans)
        val btnHistory = findViewById<Button>(R.id.btnHistory)
        val btnAbout = findViewById<Button>(R.id.btnAbout)

        cardMaize.setOnClickListener { openCamera(CROP_MAIZE) }
        cardCassava.setOnClickListener { openCamera(CROP_CASSAVA) }
        cardBeans.setOnClickListener { openCamera(CROP_BEANS) }

        btnHistory.setOnClickListener {
            startActivity(Intent(this, HistoryActivity::class.java))
        }

        btnAbout.setOnClickListener {
            AlertDialog.Builder(this)
                .setTitle("Smart AgricDiagnostics")
                .setMessage(
                    "An offline ML-based crop disease detection system.\n\n" +
                    "Crops: Maize, Cassava, Beans\n" +
                    "Model: MobileNetV2 via TensorFlow Lite\n" +
                    "Storage: Room (SQLite) — Fully Offline\n\n" +
                    "Case Study: Tororo District, Eastern Uganda\n" +
                    "Kyambogo University — 2026\n\n" +
                    "Team: Mujuni V. | Muhumuza A. | Adeke L. | Busuulwa S."
                )
                .setPositiveButton("OK", null)
                .show()
        }

        // Update history button with count
        updateHistoryCount(btnHistory)
    }

    override fun onResume() {
        super.onResume()
        updateHistoryCount(findViewById(R.id.btnHistory))
    }

    private fun updateHistoryCount(btn: Button) {
        CoroutineScope(Dispatchers.IO).launch {
            val count = AppDatabase.getDatabase(this@MainActivity).historyDao().getCount()
            withContext(Dispatchers.Main) {
                btn.text = "📋  History ($count)"
            }
        }
    }

    private fun openCamera(cropType: String) {
        val intent = Intent(this, CameraActivity::class.java)
        intent.putExtra(EXTRA_CROP, cropType)
        startActivity(intent)
    }
}
