/**
 * ---------------------------------------------------------------------------
 * Vorlesung: Deep Learning for Computer Vision (SoSe 2023)
 * Thema:     Test App for CameraX & TensorFlow Lite
 *
 * @author Jan Rexilius
 * @date   02/2023
 * ---------------------------------------------------------------------------
 */

package com.example.mindencameraapp;

import androidx.annotation.NonNull;

import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.PixelFormat;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.media.Image;
import android.util.Log;

import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.AspectRatio;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageCapture;
import androidx.camera.core.ImageCaptureException;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.core.VideoCapture;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.annotation.SuppressLint;
import android.content.ContentValues;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Size;
import android.view.View;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.google.common.util.concurrent.ListenableFuture;
import com.googlecode.tesseract.android.ResultIterator;
import com.googlecode.tesseract.android.TessBaseAPI;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;
import org.tensorflow.lite.support.image.ops.TransformToGrayscaleOp;
import org.tensorflow.lite.support.label.TensorLabel;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.MappedByteBuffer;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.ExecutorService;


// ----------------------------------------------------------------------
// main class
public class MainActivity extends AppCompatActivity implements ImageAnalysis.Analyzer {

    private static final String TAG = "LOGGING:";
    private ListenableFuture<ProcessCameraProvider> cameraProviderFuture;
    PreviewView previewView;
    ImageView tensorImageView;
    private ImageAnalysis imageAnalyzer;

    // OCR Mode 0 = single character only (single image classification)
    // 1 = single image multiple character recognition
    private int ocrMode = 0;

    ImageButton buttonTakePicture;
    ImageButton buttonGallery;
    TextView classificationResults;
    private ImageCapture imageCapture;

    private int REQUEST_CODE_PERMISSIONS = 10;

    // TODO: "android.permission.WRITE_EXTERNAL_STORAGE" is not used anymore in Android 11+, look for some other way to save files
    private final String[] REQUIRED_PERMISSIONS = new String[]{
            "android.permission.CAMERA",
    };

    private final Object task = new Object();


    private TessBaseAPI tess;

    private Interpreter tflite;
    // add your filename here (label names)
    final String CLASSIFIER_LABEL_File = "labels_feller-9.txt";
    // add your filename here (model file)
    final String TF_LITE_File = "feller-v2.tflite";
    List<String> clasifierLabels = null;




    // ----------------------------------------------------------------------
    // set gui elements and start workflow
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        classificationResults = findViewById(R.id.classificationResults);
        buttonTakePicture = findViewById(R.id.buttonCapture);
        buttonTakePicture.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                captureImage();
            }
        });

        previewView = findViewById(R.id.previewView);
        tensorImageView = findViewById(R.id.tensorImageView);

        // check permissions and start camera if all permissions available
        checkPermissions();
    }

    private boolean allPermissionsGranted() {
        for (String permission : REQUIRED_PERMISSIONS) {
            if (ContextCompat.checkSelfPermission(this, permission) != PackageManager.PERMISSION_GRANTED) {
                Log.e(TAG, "Missing permission: " + permission);
                return false;
            }
        }
        Log.i(TAG, "All good... permissions granted");
        return true;
    }

    // ----------------------------------------------------------------------
    // check app permissions
    private void checkPermissions() {
        if (allPermissionsGranted()) {
            cameraProviderFuture = ProcessCameraProvider.getInstance(this);
            cameraProviderFuture.addListener(() -> {
                try {
                    ProcessCameraProvider cameraProvider = cameraProviderFuture.get();
                    startCameraX(cameraProvider);
                } catch (ExecutionException e) {
                    e.printStackTrace();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }

            }, getExecutor());
        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS);
        }
    }

    private Executor getExecutor() {
        return ContextCompat.getMainExecutor(this);
    }


    // ----------------------------------------------------------------------
    // start camera
    @SuppressLint("RestrictedApi")
    private void startCameraX(ProcessCameraProvider cameraProvider) {
        // load label file
        try {
            clasifierLabels = FileUtil.loadLabels(this, CLASSIFIER_LABEL_File);
        } catch (IOException e) {
            Log.e("startCameraX", "Error reading label file", e);
        }
        // load tf lite model
        try{
            MappedByteBuffer tfliteModel;
            tfliteModel = FileUtil.loadMappedFile(this, TF_LITE_File);
            tflite = new Interpreter(tfliteModel);

            String path = this.getExternalFilesDir("tesseract").getAbsolutePath();
            Log.i(TAG, path);


            // Init tesseract instance with pre trained data
            tess = new TessBaseAPI();
            tess.init(path, "eng");
            tess.setPageSegMode(TessBaseAPI.PageSegMode.PSM_SINGLE_BLOCK);

        } catch (IOException e){
            Log.e("startCameraX", "Error reading model", e);
        }


        CameraSelector cameraSelector = new CameraSelector.Builder()
                .requireLensFacing(CameraSelector.LENS_FACING_BACK)
                .build();

        Preview preview = new Preview.Builder().build();
        preview.setSurfaceProvider(previewView.getSurfaceProvider());

        imageCapture = new ImageCapture.Builder()
            .setTargetResolution(new Size(2016, 1512))
            .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
            .setFlashMode(ImageCapture.FLASH_MODE_AUTO)
            .build();

        imageAnalyzer = new ImageAnalysis.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .setTargetRotation(previewView.getDisplay().getRotation())
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build();

        imageAnalyzer.setAnalyzer(getExecutor(), this);

        // unbind before binding
        cameraProvider.unbindAll();
        try {
            cameraProvider.bindToLifecycle(this, cameraSelector, imageAnalyzer, preview, imageCapture);
        } catch (Exception exc) {
            Log.e(TAG, "Use case binding failed", exc);
        }
    }


    // ----------------------------------------------------------------------
    // capture single image
    private void captureImage() {
        imageCapture.takePicture(getExecutor(), new ImageCapture.OnImageCapturedCallback() {
            @Override
            public void onCaptureSuccess(@NonNull ImageProxy image) {
                super.onCaptureSuccess(image);
                Log.d("TAG", "Capture Image");
                if(ocrMode == 0) {
                    classifySingleImage(image);
                } else if (ocrMode == 1) {
                    classifyMulti(image);
                }
            }
        });
    }


    private Bitmap rotateBitmap(Bitmap image, float degrees) {
        Matrix matrix = new Matrix();
        matrix.postRotate(degrees);
        return Bitmap.createBitmap(image, 0, 0, image.getWidth(), image.getHeight(), matrix, true);
    }

    private void classifyMulti(@NonNull ImageProxy imageProxy)  {
        ByteBuffer buffer = imageProxy.getPlanes()[0].getBuffer();
        byte[] arr = new byte[buffer.remaining()];
        buffer.get(arr);
        Bitmap bitmapImage = BitmapFactory.decodeByteArray(arr, 0, buffer.capacity());
        imageProxy.close();

        bitmapImage = rotateBitmap(bitmapImage, 90);

        int width  = bitmapImage.getWidth();
        int height = bitmapImage.getHeight();

        Log.i(TAG, "Image size: " + width + "x" + height);

        // Create bounding boxes
        tess.setImage(bitmapImage);
        tess.getUTF8Text();

        ResultIterator resultIterator = tess.getResultIterator();
        List<Rect> boxes = new ArrayList<>();

        while (resultIterator.next(TessBaseAPI.PageIteratorLevel.RIL_SYMBOL)) {
            Rect rect = resultIterator.getBoundingRect(TessBaseAPI.PageIteratorLevel.RIL_SYMBOL);
            boxes.add(rect);

            // Perform image classification
            Bitmap croppedImage = Bitmap.createBitmap(bitmapImage, rect.left, rect.top, rect.width(), rect.height());
            TensorLabel tensorLabels = predictLabel(croppedImage, false);
            Map<String, Float> floatMap = tensorLabels.getMapWithFloatValue();
            String label = getBestResult(floatMap);

            // Draw bounding box
            Canvas canvas = new Canvas(bitmapImage);
            Paint boxPaint = new Paint();
            boxPaint.setColor(Color.RED);
            boxPaint.setStyle(Paint.Style.STROKE);
            boxPaint.setStrokeWidth(5f);
            canvas.drawRect(rect, boxPaint);

            // Draw label text below the bounding box
            Paint textPaint = new Paint();
            textPaint.setColor(Color.RED);
            textPaint.setTextSize(30f);
            float textWidth = textPaint.measureText(label);
            float x = rect.left + rect.width() / 2f - textWidth / 2f;
            float y = rect.bottom + 50f; // Adjust the vertical position as needed
            canvas.drawText(label, x, y, textPaint);
        }

        // Save image
        long timeStamp = System.currentTimeMillis();
        // Save the processed bitmap to a file
        FileOutputStream outputStream;
        File processedImageFile = new File(this.getExternalFilesDir("images"), timeStamp + ".jpg"); // Change the file path and name as needed
        try {
            outputStream = new FileOutputStream(processedImageFile);
            bitmapImage.compress(Bitmap.CompressFormat.JPEG, 90, outputStream);
            outputStream.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // ----------------------------------------------------------------------
    // classify single image
    private void classifySingleImage(@NonNull ImageProxy imageProxy) {
        Log.d("classifySingleImage", "CLASSIFY_IMAGE " + imageProxy.getImageInfo().getTimestamp());
        Log.d("analyze", "format " + imageProxy.getFormat());

        ByteBuffer buffer = imageProxy.getPlanes()[0].getBuffer();
        byte[] arr = new byte[buffer.remaining()];
        buffer.get(arr);
        Bitmap bitmapImage = BitmapFactory.decodeByteArray(arr, 0, buffer.capacity());
        imageProxy.close();

        bitmapImage = rotateBitmap(bitmapImage, 90);

        String resultString = " ";
        // Map of labels and their corresponding probability
        //TensorLabel labels = new TensorLabel(clasifierLabels, probabilityProcessor.process(probabilityBuffer));
        TensorLabel labels = predictLabel(bitmapImage, true);

        // Create a map to access the result based on label
        Map<String, Float> floatMap = labels.getMapWithFloatValue();
        resultString = getResultString(floatMap);
        Log.d("classifySingleImage", "RESULT: " + resultString);
        Toast.makeText(MainActivity.this, resultString, Toast.LENGTH_SHORT).show();

        Bitmap labeledBitmap = bitmapImage.copy(Bitmap.Config.ARGB_8888, true);
        Canvas canvas = new Canvas(labeledBitmap);
        Paint paint = new Paint();
        paint.setColor(Color.BLACK);
        paint.setTextSize(100);

        Rect bounds = new Rect();
        paint.getTextBounds(resultString, 0, resultString.length(), bounds);

        int x = labeledBitmap.getWidth() / 2 - bounds.width() / 2;
        int y = labeledBitmap.getHeight() - bounds.height() - 20;

        canvas.drawText(resultString, x, y, paint);

        long timeStamp = System.currentTimeMillis();
        // Save the processed bitmap to a file
        FileOutputStream outputStream;
        File processedImageFile = new File(this.getExternalFilesDir("images"), timeStamp + ".jpg"); // Change the file path and name as needed
        try {
            outputStream = new FileOutputStream(processedImageFile);
            labeledBitmap.compress(Bitmap.CompressFormat.JPEG, 90, outputStream);
            outputStream.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private FloatBuffer normalizeByteBuffer(ByteBuffer byteBuffer) {
        FloatBuffer floatBuffer = ByteBuffer.allocateDirect(byteBuffer.capacity() * 4)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer();

        byteBuffer.rewind();
        floatBuffer.rewind();

        while (byteBuffer.hasRemaining()) {
            float val = (byteBuffer.get() & 0xFF) / 255.0f; // Normalize byte value to range [0.0, 1.0]
            floatBuffer.put(val);
        }

        floatBuffer.rewind();
        return floatBuffer;
    }

    private TensorLabel predictLabel(Bitmap image, boolean debugImage) {
        int width  = image.getWidth();
        int height = image.getHeight();

        // image size set to 73x73 (use bilinear interpolation)
        int size = height > width ? width : height;

        ImageProcessor imageProcessor = new ImageProcessor.Builder()
            .add(new ResizeWithCropOrPadOp(size, size))
            .add(new ResizeOp(73, 73, ResizeOp.ResizeMethod.BILINEAR))
            .add(new TransformToGrayscaleOp())
            //.add(new NormalizeOp(0, 255))
            .build();


        TensorImage tensorImage = new TensorImage(DataType.UINT8);
        tensorImage.load(image);
        tensorImage = imageProcessor.process(tensorImage);

        // Normalization convert uint8 [0 .. 255] to float32 [0.0 .. 1.0]
        FloatBuffer floatBuffer = normalizeByteBuffer(tensorImage.getBuffer());

        // Save the pre processed input image for the model if needed
        if(debugImage) {
            Bitmap grayscaleBitmap = Bitmap.createBitmap(73, 73, Bitmap.Config.RGB_565);
            floatBuffer.rewind(); // Reset the FloatBuffer position to start

            for (int y = 0; y < 73; y++) {
                for (int x = 0; x < 73; x++) {
                    float value = floatBuffer.get(); // Get the float value from the buffer

                    // Convert the float value to grayscale color value
                    int grayscaleColor = (int) (value * 255);
                    int pixelValue = (grayscaleColor << 16) | (grayscaleColor << 8) | grayscaleColor;

                    // Set the pixel value in the grayscale bitmap
                    grayscaleBitmap.setPixel(x, y, pixelValue);
                }
            }

            long timeStamp = System.currentTimeMillis();
            // Save the processed bitmap to a file
            FileOutputStream outputStream;
            File processedImageFile = new File(this.getExternalFilesDir("debug"), timeStamp + ".png"); // Change the file path and name as needed
            try {
                outputStream = new FileOutputStream(processedImageFile);
                grayscaleBitmap.compress(Bitmap.CompressFormat.PNG, 100, outputStream);
                outputStream.close();
            } catch (IOException e) {
                e.printStackTrace();
            }

            floatBuffer.rewind();
        }

        /*
        // Verify that the input image is float32 and all values are between 0.0 and 1.0
        while(floatBuffer.hasRemaining()) {
            float val = floatBuffer.get();
            if(val < 0.0f || val > 1.0f) {
                throw new RuntimeException("Float value " + val + " is not between 0.0 and 1.0");
            }
        }

        floatBuffer.rewind();
         */

        TensorBuffer probabilityBuffer =
            TensorBuffer.createFixedSize(new int[]{1, 62}, DataType.FLOAT32);

        if(null != tflite) {
            tflite.run(floatBuffer, probabilityBuffer.getBuffer());
        }

        return new TensorLabel(clasifierLabels, probabilityBuffer);
    }

    // ----------------------------------------------------------------------
    // process current frame
    @Override
    public void analyze(@NonNull ImageProxy imageProxy) {
        if ( imageProxy.getFormat()== PixelFormat.RGBA_8888){
            Bitmap bitmapImage = Bitmap.createBitmap(imageProxy.getWidth(),imageProxy.getHeight(),Bitmap.Config.ARGB_8888);
            bitmapImage.copyPixelsFromBuffer(imageProxy.getPlanes()[0].getBuffer());
            imageProxy.close();

            bitmapImage = rotateBitmap(bitmapImage, 90);

            // Map of labels and their corresponding probability
            //TensorLabel labels = new TensorLabel(clasifierLabels, probabilityProcessor.process(probabilityBuffer));
            TensorLabel labels = predictLabel(bitmapImage, false);

            String resultString = "";
            // Create a map to access the result based on label
            Map<String, Float> floatMap = labels.getMapWithFloatValue();
            resultString = getResultString(floatMap);
            //Log.d("classifyImage", "RESULT: " + resultString);
            classificationResults.setText(resultString);
            //Toast.makeText(MainActivity.this, resultString, Toast.LENGTH_SHORT).show();
        }
        // close image to get next one
        imageProxy.close();
    }

    // ----------------------------------------------------------------------
    // get 3 best keys & values from TF results
    public static String getResultString(Map<String, Float> mapResults){
        DecimalFormat decimalFormat = new DecimalFormat("0.00%");
        // max value
        Map.Entry<String, Float> entryMax1 = null;
        // 2nd max value
        Map.Entry<String, Float> entryMax2 = null;
        // 3rd max value
        Map.Entry<String, Float> entryMax3 = null;
        for(Map.Entry<String, Float> entry: mapResults.entrySet()){
            if (entryMax1 == null || entry.getValue().compareTo(entryMax1.getValue()) > 0){
                entryMax1 = entry;
            } else if (entryMax2 == null || entry.getValue().compareTo(entryMax2.getValue()) > 0){
                entryMax2 = entry;
            } else if (entryMax3 == null || entry.getValue().compareTo(entryMax3.getValue()) > 0){
                entryMax3 = entry;
            }
        }
        // result string includes the first three best values
        String result = entryMax1.getKey().trim() + " " + decimalFormat.format(entryMax1.getValue()) + "\n" +
                        entryMax2.getKey().trim() + " " + decimalFormat.format(entryMax2.getValue()) + "\n" +
                        entryMax3.getKey().trim() + " " + decimalFormat.format(entryMax3.getValue()) + "\n";
        return result;
    }


    // ----------------------------------------------------------------------
    // get best key & value from TF results
    public static String getBestResult(@NonNull Map<String, Float> mapResults){
        // max value
        Map.Entry<String, Float> entryMax = null;
        for(Map.Entry<String, Float> entry: mapResults.entrySet()){
            if (entryMax == null || entry.getValue().compareTo(entryMax.getValue()) > 0) {
                entryMax = entry;
            }
        }
        int val = (int)(entryMax.getValue()*100.0f);
        entryMax.setValue((float)val);
        // result string includes the first three best values
        String result = "  " + entryMax.getKey().trim() + "   (" + Integer.toString(val) + "%)";
        return result;
    }

} // class
