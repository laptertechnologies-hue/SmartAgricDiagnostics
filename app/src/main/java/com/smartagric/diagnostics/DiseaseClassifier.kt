package com.smartagric.diagnostics

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.nio.ByteBuffer
import java.nio.ByteOrder

data class DiagnosisResult(
    val disease: String,
    val confidence: Float,
    val treatment: String,
    val isDemoMode: Boolean
)

class DiseaseClassifier(private val context: Context) {

    private var interpreter: Interpreter? = null
    private var labels: List<String> = emptyList()
    private var isDemoMode = false

    private val IMAGE_SIZE = 224

    // All 9 diseases + healthy states (12 classes total) from the research report
    private val TREATMENTS = mapOf(
        "Maize_Healthy" to
            "✅ Your maize crop looks HEALTHY!\n\n" +
            "• Continue proper spacing and timely weeding\n" +
            "• Apply balanced NPK fertilizer at recommended rates\n" +
            "• Monitor regularly for early signs of disease\n" +
            "• Ensure adequate drainage to prevent waterlogging",

        "Maize_Lethal_Necrosis" to
            "🚨 URGENT — Maize Lethal Necrosis (MLN) detected!\n\n" +
            "• Immediately uproot and BURN all infected plants\n" +
            "• Do NOT compost infected material — it spreads virus\n" +
            "• Control insect vectors (aphids & thrips) with approved insecticide\n" +
            "• Plant MLN-resistant varieties (e.g., DUMA 43, H614D) next season\n" +
            "• Report outbreak to your agricultural extension officer\n" +
            "• Disinfect tools with 10% bleach solution between plants",

        "Maize_Streak_Virus" to
            "⚠️ Maize Streak Virus (MSV) detected.\n\n" +
            "• Remove and destroy severely infected plants\n" +
            "• Control leafhopper vectors with approved insecticide\n" +
            "• Plant MSV-resistant maize varieties next season\n" +
            "• Avoid late planting — early planting reduces leafhopper exposure\n" +
            "• Consult your extension officer for resistant seed varieties",

        "Maize_Northern_Leaf_Blight" to
            "⚠️ Northern Leaf Blight (NLB) detected.\n\n" +
            "• Apply fungicide: Mancozeb (2g/L) or Propiconazole per label\n" +
            "• Remove and destroy heavily infected lower leaves\n" +
            "• Ensure proper plant spacing (75cm rows) for air circulation\n" +
            "• Rotate crops — avoid planting maize in same field next season\n" +
            "• Use resistant hybrid varieties in future plantings",

        "Cassava_Healthy" to
            "✅ Your cassava crop looks HEALTHY!\n\n" +
            "• Continue regular weeding especially in first 3 months\n" +
            "• Monitor for whitefly populations (vectors of mosaic disease)\n" +
            "• Harvest at recommended maturity (8–24 months)\n" +
            "• Only use certified, disease-free planting cuttings",

        "Cassava_Mosaic_Disease" to
            "🚨 URGENT — Cassava Mosaic Disease (CMD) detected!\n\n" +
            "• Uproot and DESTROY ALL infected plants immediately\n" +
            "• Do NOT use cuttings from infected plants for planting\n" +
            "• Control whitefly vectors with approved insecticide\n" +
            "• Plant CMD-resistant varieties (e.g., NASE 14, Kiroba)\n" +
            "• Source only certified clean planting material from NARO\n" +
            "• Alert your extension officer — CMD spreads very quickly",

        "Cassava_Brown_Streak_Disease" to
            "🚨 URGENT — Cassava Brown Streak Disease (CBSD) detected!\n\n" +
            "• Remove and DESTROY all infected plants and roots\n" +
            "• NEVER replant cuttings from an infected crop\n" +
            "• Source only certified disease-free cuttings for next season\n" +
            "• Control whitefly vectors with approved insecticide\n" +
            "• Consult extension officer for CBSD-tolerant varieties\n" +
            "• Report to Tororo District Production Department",

        "Cassava_Bacterial_Blight" to
            "⚠️ Cassava Bacterial Blight (CBB) detected.\n\n" +
            "• Prune infected branches 30cm below visible lesions\n" +
            "• Disinfect cutting tools with 10% bleach between each plant\n" +
            "• Remove and burn severely infected plants\n" +
            "• Improve field drainage to reduce humidity\n" +
            "• Use only healthy, uninfected planting material\n" +
            "• Avoid working in field when plants are wet from rain/dew",

        "Beans_Healthy" to
            "✅ Your beans crop looks HEALTHY!\n\n" +
            "• Continue proper spacing (40cm between rows)\n" +
            "• Apply appropriate fertilizer — beans fix their own nitrogen\n" +
            "• Harvest pods when fully mature and dry\n" +
            "• Rotate crops — avoid planting beans in same plot for 2+ years",

        "Beans_Rust" to
            "⚠️ Bean Rust detected.\n\n" +
            "• Apply fungicide: Mancozeb (2g/L water) or copper-based fungicide\n" +
            "• Remove and destroy severely infected plant material\n" +
            "• Improve air circulation — avoid overcrowding plants\n" +
            "• Avoid overhead irrigation; water at base of plant\n" +
            "• Plant rust-resistant bean varieties in next season\n" +
            "• Begin spraying at first sign of rust pustules",

        "Beans_Angular_Leaf_Spot" to
            "⚠️ Angular Leaf Spot (ALS) detected.\n\n" +
            "• Apply copper-based fungicide (copper oxychloride 3g/L)\n" +
            "• Avoid working in field when plants are wet\n" +
            "• Remove and destroy ALL crop residue after harvest\n" +
            "• Use certified disease-free seeds for next planting\n" +
            "• Rotate crops with non-legume crops for at least 2 seasons\n" +
            "• Improve spacing to reduce leaf wetness duration",

        "Beans_Common_Bacterial_Blight" to
            "⚠️ Common Bacterial Blight (CBB) detected.\n\n" +
            "• Remove and destroy infected plant parts immediately\n" +
            "• Apply copper-based bactericide (copper hydroxide per label)\n" +
            "• Avoid overhead watering — water at soil level only\n" +
            "• Use certified disease-free seeds from trusted agro-dealer\n" +
            "• Rotate crops with cereals or non-legumes for 2+ seasons\n" +
            "• Disinfect farm tools after use in infected areas"
    )

    // Realistic demo results per crop type for DEMO MODE
    private val DEMO_RESULTS = mapOf(
        "Maize" to listOf(
            Triple("Maize_Lethal_Necrosis", 0.87f, false),
            Triple("Maize_Streak_Virus", 0.79f, false),
            Triple("Maize_Northern_Leaf_Blight", 0.91f, false),
            Triple("Maize_Healthy", 0.95f, true)
        ),
        "Cassava" to listOf(
            Triple("Cassava_Mosaic_Disease", 0.93f, false),
            Triple("Cassava_Brown_Streak_Disease", 0.82f, false),
            Triple("Cassava_Bacterial_Blight", 0.76f, false),
            Triple("Cassava_Healthy", 0.88f, true)
        ),
        "Beans" to listOf(
            Triple("Beans_Rust", 0.84f, false),
            Triple("Beans_Angular_Leaf_Spot", 0.78f, false),
            Triple("Beans_Common_Bacterial_Blight", 0.89f, false),
            Triple("Beans_Healthy", 0.92f, true)
        )
    )

    init {
        loadModel()
    }

    private fun loadModel() {
        try {
            val model = FileUtil.loadMappedFile(context, "crop_disease_model.tflite")
            val options = Interpreter.Options().apply { numThreads = 4 }
            interpreter = Interpreter(model, options)
            labels = FileUtil.loadLabels(context, "labels.txt")
            isDemoMode = false
            AppLogger.log(context, "MODEL", "TFLite model loaded successfully. ${labels.size} classes.")
        } catch (e: Exception) {
            // Model not yet trained — fall back to Demo Mode
            isDemoMode = true
            labels = FileUtil.loadLabels(context, "labels.txt")
            AppLogger.log(context, "MODEL", "TFLite model not found — DEMO MODE active. Error: ${e.message}")
        }
    }

    fun classify(bitmap: Bitmap, cropType: String): DiagnosisResult {
        AppLogger.log(context, "CLASSIFY", "Starting classification for crop: $cropType | DemoMode=$isDemoMode")

        if (isDemoMode || interpreter == null) {
            return getDemoResult(cropType)
        }

        return try {
            val scaled = Bitmap.createScaledBitmap(bitmap, IMAGE_SIZE, IMAGE_SIZE, true)
            val buffer = preprocessBitmap(scaled)
            val output = Array(1) { FloatArray(labels.size) }
            interpreter!!.run(buffer, output)

            val probs = output[0]
            val maxIdx = probs.indices.maxByOrNull { probs[it] } ?: 0
            val maxConf = probs[maxIdx]
            val disease = labels[maxIdx]
            val treatment = TREATMENTS[disease]
                ?: "Consult your local agricultural extension officer for advice."

            AppLogger.log(context, "RESULT", "Disease=$disease | Confidence=${(maxConf*100).toInt()}%")
            DiagnosisResult(disease, maxConf, treatment, false)
        } catch (e: Exception) {
            AppLogger.log(context, "ERROR", "Inference failed: ${e.message}. Falling back to demo.")
            getDemoResult(cropType)
        }
    }

    private fun getDemoResult(cropType: String): DiagnosisResult {
        val options = DEMO_RESULTS[cropType] ?: DEMO_RESULTS["Maize"]!!
        // Simulate inference by picking a weighted-random result
        val pick = options[(System.currentTimeMillis() % options.size).toInt()]
        val disease   = pick.first
        val confidence = pick.second
        val treatment = TREATMENTS[disease]
            ?: "Consult your local agricultural extension officer for advice."
        AppLogger.log(context, "DEMO", "Returning demo result: $disease (${(confidence*100).toInt()}%)")
        return DiagnosisResult(disease, confidence, treatment, isDemoMode = true)
    }

    private fun preprocessBitmap(bitmap: Bitmap): ByteBuffer {
        val buf = ByteBuffer.allocateDirect(4 * IMAGE_SIZE * IMAGE_SIZE * 3)
        buf.order(ByteOrder.nativeOrder())
        val pixels = IntArray(IMAGE_SIZE * IMAGE_SIZE)
        bitmap.getPixels(pixels, 0, IMAGE_SIZE, 0, 0, IMAGE_SIZE, IMAGE_SIZE)
        for (pixel in pixels) {
            // MobileNetV2 normalisation: [-1, 1]
            buf.putFloat(((pixel shr 16 and 0xFF) - 127.5f) / 127.5f)
            buf.putFloat(((pixel shr 8  and 0xFF) - 127.5f) / 127.5f)
            buf.putFloat(((pixel        and 0xFF) - 127.5f) / 127.5f)
        }
        return buf
    }

    fun close() { interpreter?.close() }
}
