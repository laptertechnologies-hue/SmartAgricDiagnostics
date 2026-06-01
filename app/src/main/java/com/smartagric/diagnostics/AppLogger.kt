package com.smartagric.diagnostics

import android.content.Context
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * AppLogger — offline event logger for Smart AgricDiagnostics.
 *
 * Writes timestamped log entries to a local file:
 *   /data/data/com.smartagric.diagnostics/files/agric_diagnostics.log
 *
 * This log is used to audit diagnoses, track demo mode usage, and debug
 * the ML inference pipeline without requiring internet connectivity.
 */
object AppLogger {

    private const val LOG_FILE_NAME = "agric_diagnostics.log"
    private const val MAX_LOG_SIZE_BYTES = 512_000L // 512 KB cap

    fun log(context: Context, tag: String, message: String) {
        try {
            val logFile = File(context.filesDir, LOG_FILE_NAME)

            // Rotate log if it exceeds 512 KB
            if (logFile.exists() && logFile.length() > MAX_LOG_SIZE_BYTES) {
                val archive = File(context.filesDir, "agric_diagnostics_old.log")
                logFile.copyTo(archive, overwrite = true)
                logFile.delete()
            }

            val timestamp = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault())
                .format(Date())
            val entry = "[$timestamp] [$tag] $message\n"
            logFile.appendText(entry)
        } catch (e: Exception) {
            // Silent fail — logging must never crash the app
        }
    }

    fun getLogContent(context: Context): String {
        return try {
            File(context.filesDir, LOG_FILE_NAME).readText()
        } catch (e: Exception) {
            "No log entries yet."
        }
    }

    fun clearLog(context: Context) {
        try {
            File(context.filesDir, LOG_FILE_NAME).delete()
        } catch (e: Exception) { /* ignore */ }
    }
}
