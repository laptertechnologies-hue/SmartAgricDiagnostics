package com.smartagric.diagnostics.data

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.Query

@Dao
interface HistoryDao {

    @Query("SELECT * FROM diagnosis_history ORDER BY timestamp DESC")
    fun getAllHistory(): List<DiagnosisRecord>

    @Insert
    fun insert(record: DiagnosisRecord)

    @Query("SELECT COUNT(*) FROM diagnosis_history")
    fun getCount(): Int

    @Query("DELETE FROM diagnosis_history")
    fun clearAll()
}
