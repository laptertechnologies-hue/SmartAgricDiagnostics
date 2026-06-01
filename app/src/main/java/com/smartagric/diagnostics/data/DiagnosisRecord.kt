package com.smartagric.diagnostics.data

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "diagnosis_history")
data class DiagnosisRecord(
    @PrimaryKey(autoGenerate = true) val id: Int = 0,
    val crop: String,
    val disease: String,
    val confidence: Float,
    val treatment: String,
    val isDemo: Boolean = false,
    val timestamp: Long = System.currentTimeMillis()
)
