package com.smartagric.diagnostics

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import com.smartagric.diagnostics.data.DiagnosisRecord
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class HistoryAdapter(private val records: List<DiagnosisRecord>) :
    RecyclerView.Adapter<HistoryAdapter.ViewHolder>() {

    class ViewHolder(view: View) : RecyclerView.ViewHolder(view) {
        val emoji      : TextView = view.findViewById(R.id.tvItemEmoji)
        val crop       : TextView = view.findViewById(R.id.tvItemCrop)
        val disease    : TextView = view.findViewById(R.id.tvItemDisease)
        val date       : TextView = view.findViewById(R.id.tvItemDate)
        val confidence : TextView = view.findViewById(R.id.tvItemConfidence)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_history, parent, false)
        return ViewHolder(view)
    }

    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        val record = records[position]
        val fmt = SimpleDateFormat("dd MMM yyyy, HH:mm", Locale.getDefault())

        holder.emoji.text = when (record.crop) {
            "Maize"   -> "🌽"
            "Cassava" -> "🌿"
            "Beans"   -> "🫘"
            else      -> "🌱"
        }
        holder.crop.text       = record.crop.uppercase()
        holder.disease.text    = record.disease
        holder.date.text       = fmt.format(Date(record.timestamp))
        val pct = (record.confidence * 100).toInt()
        holder.confidence.text = "$pct%"

        val isHealthy = record.disease.contains("Healthy", ignoreCase = true)
        val color = if (isHealthy)
            holder.confidence.context.getColor(R.color.healthy_color)
        else
            holder.confidence.context.getColor(R.color.disease_color)
        holder.confidence.setTextColor(color)
    }

    override fun getItemCount() = records.size
}
