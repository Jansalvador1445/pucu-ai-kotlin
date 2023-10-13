package com.example.demotensorflow

import android.content.Intent
import android.graphics.Bitmap
import android.net.Uri
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.provider.MediaStore
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import com.example.demotensorflow.ml.MobilenetV110224Quant
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer

class MainActivity : AppCompatActivity() {

    lateinit var imageView: ImageView
    lateinit var bitmap: Bitmap

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        imageView = findViewById(R.id.imageView2)


        val fileName = "labels.txt"
        val inputString = application.assets.open(fileName).bufferedReader().use {
            it.readText()
        }
        var townList = inputString.split("\n")

        var textView: TextView = findViewById(R.id.textView)
        var select: Button = findViewById(R.id.select)

        select.setOnClickListener(View.OnClickListener {
            var intent: Intent = Intent(Intent.ACTION_GET_CONTENT)
            intent.type ="image/*"
            startActivityForResult(intent, 100)
        })

        var predict: Button = findViewById(R.id.predict)
        predict.setOnClickListener(View.OnClickListener {
            var resize: Bitmap = Bitmap.createScaledBitmap(
                bitmap,
                224,
                224,
                true
            )
            val model = MobilenetV110224Quant.newInstance(this)
            var theBuffer = TensorImage.fromBitmap(resize)
            var byteBuffer = theBuffer.buffer

            val inputFeature0 = TensorBuffer.createFixedSize(
                intArrayOf(1, 224, 224, 3),
                DataType.UINT8
            )
            inputFeature0.loadBuffer(byteBuffer)

            val outputs = model.process(inputFeature0)
            val outputFeature0 = outputs.outputFeature0AsTensorBuffer
            var max = getMax(outputFeature0.floatArray)
            textView.setText(townList[max])

            model.close()
        })
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        imageView.setImageURI(data?.data)
        var uri: Uri? = data?.data
        bitmap = MediaStore.Images.Media.getBitmap(
            this.contentResolver,
            uri
        )
    }

//   Get Max
    fun getMax(arr: FloatArray): Int {
        var index = 0
        var min = 0.0f

        for(i in 0..1000){
            if(arr[i]>min){
                index = i
                min = arr[i]
            }
        }
        return index
    }
}