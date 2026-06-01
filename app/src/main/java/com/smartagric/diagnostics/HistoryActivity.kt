package com.smartagric.diagnostics

import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.smartagric.diagnostics.data.AppDatabase
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class HistoryActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_history)

        findViewById<Button>(R.id.btnBack).setOnClickListener { finish() }

        loadHistory()
    }

    private fun loadHistory() {
        CoroutineScope(Dispatchers.IO).launch {
            val records = AppDatabase.getDatabase(this@HistoryActivity).historyDao().getAllHistory()
            withContext(Dispatchers.Main) {
                val recycler  = findViewById<RecyclerView>(R.id.recyclerHistory)
                val emptyView = findViewById<View>(R.id.emptyState)
                val tvCount   = findViewById<TextView>(R.id.tvCount)

                tvCount.text = "${records.size} record${if (records.size != 1) "s" else ""}"

                if (records.isEmpty()) {
                    recycler.visibility  = View.GONE
                    emptyView.visibility = View.VISIBLE
                } else {
                    recycler.visibility  = View.VISIBLE
                    emptyView.visibility = View.GONE
                    recycler.layoutManager = LinearLayoutManager(this@HistoryActivity)
                    recycler.adapter = HistoryAdapter(records)
                }
            }
        }
    }
}
