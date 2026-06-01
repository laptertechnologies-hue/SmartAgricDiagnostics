package com.smartagric.diagnostics

import android.content.Intent
import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.smartagric.diagnostics.data.AppDatabase
import com.smartagric.diagnostics.data.DiagnosisRecord
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class ResultActivity : AppCompatActivity() {

    private lateinit var cropType: String
    private lateinit var diseaseName: String
    private var confidence: Float = 0f
    private lateinit var treatment: String
    private var isDemoMode: Boolean = false
    private var isSaved = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_result)

        cropType    = intent.getStringExtra(MainActivity.EXTRA_CROP) ?: "Maize"
        diseaseName = intent.getStringExtra(CameraActivity.EXTRA_DISEASE) ?: "Unknown"
        confidence  = intent.getFloatExtra(CameraActivity.EXTRA_CONFIDENCE, 0f)
        treatment   = intent.getStringExtra(CameraActivity.EXTRA_TREATMENT) ?: ""
        isDemoMode  = intent.getBooleanExtra(CameraActivity.EXTRA_IS_DEMO, false)

        bindViews()
        setupButtons()
        AppLogger.log(this, "RESULT", "Crop=$cropType | Disease=$diseaseName | Conf=${(confidence*100).toInt()}% | Demo=$isDemoMode")
    }

    private fun bindViews() {
        val isHealthy = diseaseName.contains("Healthy", ignoreCase = true)
        val confidencePct = (confidence * 100).toInt()

        // Demo banner
        val demoBanner = findViewById<TextView>(R.id.tvDemoBanner)
        demoBanner.visibility = if (isDemoMode) View.VISIBLE else View.GONE

        // Crop emoji
        val emoji = when (cropType) {
            "Maize"   -> "🌽"
            "Cassava" -> "🌿"
            "Beans"   -> "🫘"
            else      -> "🌱"
        }
        findViewById<TextView>(R.id.tvCropEmoji).text = emoji
        findViewById<TextView>(R.id.tvCropName).text  = cropType.uppercase()

        // Status badge
        val badge = findViewById<TextView>(R.id.tvStatusBadge)
        if (isHealthy) {
            badge.text = "✅  HEALTHY"
            badge.setBackgroundColor(ContextCompat.getColor(this, R.color.healthy_color))
        } else {
            badge.text = "⚠️  DISEASE DETECTED"
            badge.setBackgroundColor(ContextCompat.getColor(this, R.color.disease_color))
        }

        // Disease name — clean up underscore formatting
        val cleanName = diseaseName.replace("_", " ")
        findViewById<TextView>(R.id.tvDiseaseName).text = cleanName

        // Confidence bar
        val bar = findViewById<ProgressBar>(R.id.confidenceBar)
        bar.progress = confidencePct
        val barColor = if (isHealthy) R.color.healthy_color else R.color.disease_color
        bar.progressTintList = ContextCompat.getColorStateList(this, barColor)
        findViewById<TextView>(R.id.tvConfidenceValue).text = "$confidencePct% confident"

        // Treatment
        findViewById<TextView>(R.id.tvTreatment).text = treatment
    }

    private fun setupButtons() {
        findViewById<Button>(R.id.btnBack).setOnClickListener { finish() }

        findViewById<Button>(R.id.btnSave).setOnClickListener {
            if (isSaved) {
                Toast.makeText(this, "Already saved!", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            val record = DiagnosisRecord(
                crop       = cropType,
                disease    = diseaseName.replace("_", " "),
                confidence = confidence,
                treatment  = treatment,
                isDemo     = isDemoMode
            )
            CoroutineScope(Dispatchers.IO).launch {
                AppDatabase.getDatabase(this@ResultActivity).historyDao().insert(record)
                AppLogger.log(this@ResultActivity, "SAVED", "Record saved: ${record.disease}")
                withContext(Dispatchers.Main) {
                    isSaved = true
                    Toast.makeText(this@ResultActivity, "✅ Diagnosis saved to history!", Toast.LENGTH_SHORT).show()
                    findViewById<Button>(R.id.btnSave).text = "✅  Saved!"
                }
            }
        }

        findViewById<Button>(R.id.btnScanAgain).setOnClickListener {
            val intent = Intent(this, CameraActivity::class.java)
            intent.putExtra(MainActivity.EXTRA_CROP, cropType)
            startActivity(intent)
            finish()
        }
    }
}
