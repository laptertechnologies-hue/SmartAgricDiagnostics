package com.smartagric.diagnostics

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import java.nio.ByteBuffer
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class CameraActivity : AppCompatActivity() {

    private var imageCapture: ImageCapture? = null
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var classifier: DiseaseClassifier
    private lateinit var cropType: String

    companion object {
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
        const val EXTRA_DISEASE = "extra_disease"
        const val EXTRA_CONFIDENCE = "extra_confidence"
        const val EXTRA_TREATMENT = "extra_treatment"
        const val EXTRA_IS_DEMO = "extra_is_demo"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_camera)

        cropType = intent.getStringExtra(MainActivity.EXTRA_CROP) ?: "Maize"

        findViewById<TextView>(R.id.tvCropName).text = cropType
        classifier = DiseaseClassifier(this)

        cameraExecutor = Executors.newSingleThreadExecutor()

        if (allPermissionsGranted()) startCamera()
        else ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)

        findViewById<Button>(R.id.btnBack).setOnClickListener { finish() }
        findViewById<Button>(R.id.btnCapture).setOnClickListener { takePhotoAndDiagnose() }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(findViewById<PreviewView>(R.id.viewFinder).surfaceProvider)
            }
            imageCapture = ImageCapture.Builder()
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                .build()
            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, imageCapture)
            } catch (e: Exception) {
                Toast.makeText(this, "Camera failed to start", Toast.LENGTH_SHORT).show()
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun takePhotoAndDiagnose() {
        val imageCapture = imageCapture ?: return
        val captureBtn = findViewById<Button>(R.id.btnCapture)
        val analyzingText = findViewById<TextView>(R.id.tvAnalyzing)

        captureBtn.isEnabled = false
        captureBtn.text = "Analyzing..."
        analyzingText.visibility = View.VISIBLE
        analyzingText.text = "🔬 Running AI diagnosis..."

        imageCapture.takePicture(ContextCompat.getMainExecutor(this), object : ImageCapture.OnImageCapturedCallback() {
            override fun onCaptureSuccess(image: ImageProxy) {
                val bitmap = imageProxyToBitmap(image)
                image.close()

                val result = classifier.classify(bitmap, cropType)

                runOnUiThread {
                    analyzingText.visibility = View.GONE
                    captureBtn.isEnabled = true
                    captureBtn.text = "🔬  DIAGNOSE DISEASE"

                    val intent = Intent(this@CameraActivity, ResultActivity::class.java).apply {
                        putExtra(MainActivity.EXTRA_CROP, cropType)
                        putExtra(EXTRA_DISEASE, result.disease)
                        putExtra(EXTRA_CONFIDENCE, result.confidence)
                        putExtra(EXTRA_TREATMENT, result.treatment)
                        putExtra(EXTRA_IS_DEMO, result.isDemoMode)
                    }
                    startActivity(intent)
                }
            }

            override fun onError(exception: ImageCaptureException) {
                runOnUiThread {
                    analyzingText.visibility = View.GONE
                    captureBtn.isEnabled = true
                    captureBtn.text = "🔬  DIAGNOSE DISEASE"
                    Toast.makeText(this@CameraActivity, "Capture failed. Try again.", Toast.LENGTH_SHORT).show()
                }
            }
        })
    }

    private fun imageProxyToBitmap(image: ImageProxy): Bitmap {
        val buffer: ByteBuffer = image.planes[0].buffer
        val bytes = ByteArray(buffer.remaining())
        buffer.get(bytes)
        return BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
            ?: Bitmap.createBitmap(224, 224, Bitmap.Config.ARGB_8888)
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) startCamera()
            else {
                Toast.makeText(this, "Camera permission denied", Toast.LENGTH_SHORT).show()
                finish()
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        classifier.close()
    }
}
