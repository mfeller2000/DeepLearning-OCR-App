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

import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.PixelFormat;
import android.graphics.Rect;
import android.net.Uri;
import android.os.Environment;
import android.util.Log;

import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.AspectRatio;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageCapture;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.core.content.FileProvider;

import android.annotation.SuppressLint;
import android.os.Bundle;
import android.util.Size;
import android.view.View;
import android.widget.CheckBox;
import android.widget.EditText;
import android.widget.ImageButton;
import android.widget.TextView;
import android.widget.Toast;

import com.google.common.util.concurrent.ListenableFuture;
import com.googlecode.tesseract.android.ResultIterator;
import com.googlecode.tesseract.android.TessBaseAPI;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.Operator;
import org.tensorflow.lite.support.common.TensorOperator;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;
import org.tensorflow.lite.support.image.ops.TransformToGrayscaleOp;
import org.tensorflow.lite.support.label.TensorLabel;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.MappedByteBuffer;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;


// ----------------------------------------------------------------------
// main class
public class MainActivity extends AppCompatActivity implements ImageAnalysis.Analyzer {

    private static final String TAG = "LOGGING:";
    private ListenableFuture<ProcessCameraProvider> cameraProviderFuture;
    PreviewView previewView;
    private ImageAnalysis imageAnalyzer;

    ImageButton buttonTakePicture;
    TextView classificationResults;
    CheckBox ocrModeCheckbox;
    CheckBox debugImageCheckbox;

    EditText thresholdTextBox;
    private ImageCapture imageCapture;

    private int REQUEST_CODE_PERMISSIONS = 10;

    private final String[] REQUIRED_PERMISSIONS = new String[]{
            "android.permission.CAMERA",
    };

    private final Object task = new Object();


    private TessBaseAPI tess;

    private Interpreter tflite;
    // add your filename here (label names)
    final String CLASSIFIER_LABEL_File = "labels_feller-9.txt";
    // add your filename here (model file)
    final String TF_LITE_File = "feller-v2-student.tflite";

    final String TESSERACT_File = "eng.traineddata";
    List<String> clasifierLabels = null;

    final Size TARGET_RESOLUTION = new Size(1080, 1920);


    // ----------------------------------------------------------------------
    // set gui elements and start workflow
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        classificationResults = findViewById(R.id.classificationResults);
        buttonTakePicture = findViewById(R.id.buttonCapture);
        thresholdTextBox = findViewById(R.id.editThresholdText);
        buttonTakePicture.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                captureImage();
            }
        });

        previewView = findViewById(R.id.previewView);
        ocrModeCheckbox = findViewById(R.id.ocrModeCheckbox);
        debugImageCheckbox = findViewById(R.id.debugImageCheckbox);

        thresholdTextBox.setVisibility(ocrModeCheckbox.isChecked() ? View.VISIBLE : View.INVISIBLE);
        thresholdTextBox.setText("128");
        ocrModeCheckbox.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                thresholdTextBox.setVisibility(ocrModeCheckbox.isChecked() ? View.VISIBLE : View.INVISIBLE);
            }
        });

        // check permissions and start camera if all permissions available
        checkPermissions();
    }

    private String getAppFolderPath(String folder) {
        return this.getExternalFilesDir(folder).getAbsolutePath();
    }

    private boolean allPermissionsGranted() {
        for (String permission : REQUIRED_PERMISSIONS) {
            if (ContextCompat.checkSelfPermission(this, permission) != PackageManager.PERMISSION_GRANTED) {
                Log.e(TAG, "Missing permission: " + permission);
                return false;
            }
        }
        Log.d(TAG, "All good... permissions granted");
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


    /*
     * Tesseract requires the files to be placed into internal storage under s specific path
     * that's why we need to check if the file exists, if not copy it from the assets to the
     * required path
     */
    private void checkTessdata() {
        AssetManager assetManager = this.getAssets();
        String path = getAppFolderPath("tesseract") + "/tessdata";
        InputStream tessdata = null;

        File copiedFile = new File(path, TESSERACT_File);

        if(copiedFile.exists()) {
            Log.d("tesseract", "Found tesseract file.. All good no need for copy");
            return;
        }

        try {
            tessdata = assetManager.open(TESSERACT_File);
        } catch (IOException e) {
            Log.e("tesseract", "Failed to get Tessdata file", e);
        }

        try {
            // Create directories if they don't exist
            Files.createDirectories(Paths.get(path));

            OutputStream outputStream = new FileOutputStream(copiedFile);

            // copy file from assets to internal storage
            copyFile(tessdata, outputStream);
            tessdata.close();
            outputStream.flush();
            outputStream.close();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

    }

    /*
    * Helper function for copying files
    */
    private void copyFile(InputStream in, OutputStream out) throws IOException {
        byte[] buffer = new byte[1024];
        int read;
        while((read = in.read(buffer)) != -1){
            out.write(buffer, 0, read);
        }
    }

    /*
     * Start camera
     */
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
            // Load Tensorflow model
            tfliteModel = FileUtil.loadMappedFile(this, TF_LITE_File);
            tflite = new Interpreter(tfliteModel);

            // check if tesseract file exists else copy to internal storage
            checkTessdata();
            String path = getAppFolderPath("tesseract");


            tess = new TessBaseAPI();
            // Segmentation mode, consider whole image a possible text
            tess.setPageSegMode(TessBaseAPI.PageSegMode.PSM_SINGLE_BLOCK);
            // set minimum heights for possible characters
            tess.setVariable("textord_old_xheight", "1");
            tess.setVariable("textord_min_xheight", "30");

            // Init tesseract instance
            tess.init(path, "eng");

        } catch (IOException e){
            Log.e("startCameraX", "Error reading model", e);
        }


        CameraSelector cameraSelector = new CameraSelector.Builder()
                .requireLensFacing(CameraSelector.LENS_FACING_BACK)
                .build();

        Preview preview = new Preview.Builder().build();
        preview.setSurfaceProvider(previewView.getSurfaceProvider());

        imageCapture = new ImageCapture.Builder()
            .setTargetResolution(TARGET_RESOLUTION)
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
                if(ocrModeCheckbox.isChecked()) {
                    classifyMulti(image);
                } else {
                    classifySingleImage(image);
                }
            }
        });
    }


    /*
    * Rotates a bitmap image by degrees
    * */
    private Bitmap rotateBitmap(Bitmap image, float degrees) {
        Matrix matrix = new Matrix();
        matrix.postRotate(degrees);
        return Bitmap.createBitmap(image, 0, 0, image.getWidth(), image.getHeight(), matrix, true);
    }

    /*
    * Binarize a bitmap image, set pixel to to BLACK if below threshold or to WHITE if above threshold
    * */
    private Bitmap binarizeBitmap(Bitmap image, int threshold) {
        for(int x = 0; x < image.getWidth(); x++) {
            for(int y = 0; y < image.getHeight(); y++) {
                int pixel = image.getPixel(x, y);

                // Extract the color channels (assuming it's a grayscale image)
                int red = (pixel >> 16) & 0xff;
                int green = (pixel >> 8) & 0xff;
                int blue = pixel & 0xff;

                // Calculate the average value of the color channels
                int average = (red + green + blue) / 3;

                // Compare the average value with the threshold
                int binaryPixel = (average < threshold) ? Color.BLACK : Color.WHITE;

                image.setPixel(x, y, binaryPixel);
            }
        }

        return image;
    }

    /*
    * Classify multiple characters with a single image
    * */
    private void classifyMulti(@NonNull ImageProxy imageProxy)  {
        // get bitmap image
        Bitmap bitmapImage = getBitmapImage(imageProxy);

        int width  = bitmapImage.getWidth();
        int height = bitmapImage.getHeight();

        Log.d("classifyMulti", "Image size: " + width + "x" + height);

        int threshold = Integer.parseInt(thresholdTextBox.getText().toString());

        if(threshold < 0 || threshold > 255) {
            Toast.makeText(MainActivity.this, "Threshold not in range (0 .. 255)", Toast.LENGTH_LONG).show();
            Log.e("classifyMulti", "Invalid threshold values " + threshold);
            return;
        }

        // Create a new bitmap for drawing bounding boxes and labels
        Bitmap imageResult = bitmapImage.copy(bitmapImage.getConfig(), true);

        // Create bounding boxes
        tess.setImage(bitmapImage);
        // Start Tesseract OCR mode, this will also get the characters but we don't need
        // them because we only need the bounding boxes
        tess.getUTF8Text();


        ResultIterator resultIterator = tess.getResultIterator();
        List<Rect> boxes = new ArrayList<>();

        // Iterate over all found bounding boxes and add them to a list
        // PageIteratorLevel.RIL_SYMBOL = Create bounding boxes arond Symbols/Characters
        while (resultIterator.next(TessBaseAPI.PageIteratorLevel.RIL_SYMBOL)) {
            Rect rect = resultIterator.getBoundingRect(TessBaseAPI.PageIteratorLevel.RIL_SYMBOL);
            boxes.add(rect);
        }

        // Abort prediction if there are too many bounding boxes.. avoids hang up..
        if(boxes.size() > 250) {
            Log.e("classifyMulti", "Reached hard limit of max. amount of bounding boxes.. refusing to predict");
            Toast.makeText(MainActivity.this, "Didn't save.. too much to predict", Toast.LENGTH_LONG).show();
            return;
        }

        String predictedText = "";

        // Iterate over all bounding boxes and send each box to the tensorflow model
        for (Rect rect : boxes) {
            // Crop image with bounding box
            Bitmap croppedImage = Bitmap.createBitmap(bitmapImage, rect.left, rect.top, rect.width(), rect.height());

            // Get longes side of cropped image
            int maxSize = Math.max(croppedImage.getWidth(), croppedImage.getHeight());
            // Create empty squared image
            Bitmap squareImage = Bitmap.createBitmap(maxSize, maxSize, Bitmap.Config.ARGB_8888);

            // Set image to white
            Canvas squareCanvas = new Canvas(squareImage);
            squareCanvas.drawColor(Color.WHITE); // Fill the entire square image with 0 values

            // Calculate top and left core for image placement
            int left = (maxSize - croppedImage.getWidth()) / 2;
            int top = (maxSize - croppedImage.getHeight()) / 2;
            // Place cropped image into squared image
            squareCanvas.drawBitmap(croppedImage, left, top, null);



            // Binarize the squared image
            squareImage = binarizeBitmap(squareImage, threshold);

            // Predict label
            TensorLabel tensorLabels = predictLabelSingle(squareImage);

            // Convert results
            Map<String, Float> floatMap = tensorLabels.getMapWithFloatValue();
            String label = getBestResult(floatMap);
            float confidence = getBest(floatMap);

            int color = Color.RED;

            // Set the bounding box and text color of how confident the predictions was
            // above 95% green
            if(confidence > 95.0f) {
                color = 0xFF00A300;
            // above 75% yellow
            } else if (confidence > 75.0f) {
                color = Color.YELLOW;
            // above 50% orange
            } else if (confidence > 50.0f) {
                color = 0xFFFFA500;
            }
            // everything else (<=50%) keep it red

            // Draw bounding box
            Canvas canvas = new Canvas(imageResult);
            Paint boxPaint = new Paint();
            boxPaint.setColor(color);
            boxPaint.setStyle(Paint.Style.STROKE);
            boxPaint.setStrokeWidth(2f);
            canvas.drawRect(rect, boxPaint);

            // Draw label text below the bounding box
            Paint textPaint = new Paint();
            textPaint.setColor(color);
            textPaint.setTextSize(20f);
            float textWidth = textPaint.measureText(label);
            float x = rect.left + rect.width() / 2f - textWidth / 2f;
            float y = rect.bottom + 15f;
            canvas.drawText(label, x, y, textPaint);

            predictedText += label;
        }

        Log.d("classifyMulti", "Prediction: " + predictedText);

        // save image to internal storage
        saveImage(imageResult, "images", true);
    }

    private void saveImage(Bitmap image, String folder, boolean openViewer) {
        long timeStamp = System.currentTimeMillis();
        FileOutputStream outputStream;
        String fileName = timeStamp + ".jpg";

        String path = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES).getAbsolutePath() + "/DeepLearningApp/" + folder;

        // Create folder path if it doesn't exists
        try {
            Files.createDirectories(Paths.get(path));
        } catch (IOException e) {
            Log.e("saveImage", "Error creating/finding folder path", e);
            return;
        }

        String fullFilePath = path  + "/" + fileName;

        // Save the image to internal storage
        File processedImageFile = new File(fullFilePath);
        try {
            outputStream = new FileOutputStream(processedImageFile);
            image.compress(Bitmap.CompressFormat.JPEG, 90, outputStream);
            outputStream.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

        Log.d("saveImage", "Saved image to " + path);
        Toast.makeText(MainActivity.this, "Image saved", Toast.LENGTH_SHORT).show();

        // Open viewer to view the image
        if(openViewer) {
            Uri photoURI = FileProvider.getUriForFile(this, this.getApplicationContext().getPackageName() + ".provider", processedImageFile);

            Intent intent = new Intent();
            intent.setAction(Intent.ACTION_VIEW);
            intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
            intent.setDataAndType(photoURI, "image/*");
            startActivity(intent);
        }
    }

    /*
     * Get bitmap image from the image proxy
     */
    private Bitmap getBitmapImage(@NonNull ImageProxy imageProxy) {
        ByteBuffer buffer = imageProxy.getPlanes()[0].getBuffer();
        byte[] arr = new byte[buffer.remaining()];
        buffer.get(arr);
        Bitmap bitmapImage = BitmapFactory.decodeByteArray(arr, 0, buffer.capacity());
        imageProxy.close();

        // rotate
        bitmapImage = rotateBitmap(bitmapImage, 90);

        return bitmapImage;
    }

    /*
     * classify single image (single character only)
     */
    private void classifySingleImage(@NonNull ImageProxy imageProxy) {
        Log.d("classifySingleImage", "CLASSIFY_IMAGE " + imageProxy.getImageInfo().getTimestamp());
        Log.d("analyze", "format " + imageProxy.getFormat());

        // get bitmap image
        Bitmap bitmapImage = getBitmapImage(imageProxy);

        int width  = bitmapImage.getWidth();
        int height = bitmapImage.getHeight();

        Log.d("classifySingleImage", "Image size: " + width + "x" + height);

        String resultString = " ";
        // Map of labels and their corresponding probability
        TensorLabel labels = predictLabelSingle(bitmapImage);

        // Create a map to access the result based on label
        Map<String, Float> floatMap = labels.getMapWithFloatValue();
        resultString = getResultString(floatMap);
        Log.d("classifySingleImage", "RESULT: " + resultString);
        Toast.makeText(MainActivity.this, resultString, Toast.LENGTH_SHORT).show();

        // Write results as text at the bottom of bitmap image
        Bitmap labeledBitmap = bitmapImage.copy(Bitmap.Config.ARGB_8888, true);
        Canvas canvas = new Canvas(labeledBitmap);
        Paint paint = new Paint();
        paint.setColor(Color.BLACK);
        paint.setTextSize(50);


        Rect bounds = new Rect();
        paint.getTextBounds(resultString, 0, resultString.length(), bounds);

        // calculate positions of where text needs to be placed
        int x = labeledBitmap.getWidth() / 2 - bounds.width() / 2;
        int y = labeledBitmap.getHeight() - bounds.height() - 20;

        canvas.drawText(resultString, x, y, paint);

        // save image
        saveImage(labeledBitmap, "images", true);
    }

    /*
     * Normalize the buffer from byte to float 32 [0.0 .. 1.0]
     */
    private FloatBuffer normalizeByteBuffer(ByteBuffer byteBuffer) {
        // Allocate buffer, original buffer size * 4 (32 bit)
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


    private TensorLabel predictLabelSingle(Bitmap image) {
        return predictLabel(image, true);
    }

    private TensorLabel predictLabelStream(Bitmap image) {
        // disable debug image saving completely, else debug folder gets flooded with images (if checked)
        return predictLabel(image, false);
    }

    private TensorLabel predictLabel(Bitmap image, boolean isSingle) {
        int width  = image.getWidth();
        int height = image.getHeight();

        // image size set to 73x73 (use bilinear interpolation)
        int size = height > width ? width : height;

        // Preprocessing
        ImageProcessor imageProcessor = new ImageProcessor.Builder()
            .add(new ResizeWithCropOrPadOp(size, size))
            .add(new ResizeOp(73, 73, ResizeOp.ResizeMethod.BILINEAR))
            .add(new TransformToGrayscaleOp())
            .build();

        // create tensorimage
        TensorImage tensorImage = new TensorImage(DataType.UINT8);
        tensorImage.load(image);
        tensorImage = imageProcessor.process(tensorImage);

        // normalization convert uint8 [0 .. 255] to float32 [0.0 .. 1.0]
        FloatBuffer floatBuffer = normalizeByteBuffer(tensorImage.getBuffer());

        // Save the input image to the debug folder if checked and is only a single image
        if(debugImageCheckbox.isChecked() && isSingle) {
            Bitmap grayscaleBitmap = Bitmap.createBitmap(73, 73, Bitmap.Config.RGB_565);
            floatBuffer.rewind();

            // convert float values back to byte values
            for (int y = 0; y < 73; y++) {
                for (int x = 0; x < 73; x++) {
                    float value = floatBuffer.get();

                    // Convert the float value to grayscale color value
                    int grayscaleColor = (int) (value * 255);
                    int pixelValue = (grayscaleColor << 16) | (grayscaleColor << 8) | grayscaleColor;

                    // Set the pixel value in the grayscale bitmap
                    grayscaleBitmap.setPixel(x, y, pixelValue);
                }
            }

            // save debug image
            saveImage(grayscaleBitmap, "debug", false);
            floatBuffer.rewind();
        }

        TensorBuffer probabilityBuffer =
            TensorBuffer.createFixedSize(new int[]{1, 62}, DataType.FLOAT32);

        // run model
        if(null != tflite) {
            tflite.run(floatBuffer, probabilityBuffer.getBuffer());
        }

        // Apply softmax because model outputs only logits
        float[] probabilities = probabilityBuffer.getFloatArray();
        softmax(probabilities);
        probabilityBuffer.loadArray(probabilities);

        return new TensorLabel(clasifierLabels, probabilityBuffer);
    }

    /*
     * Softmax function
     */
    private void softmax(float[] input) {
        float max = Float.NEGATIVE_INFINITY;
        for (float value : input) {
            if (value > max) {
                max = value;
            }
        }

        float sum = 0.0f;
        for (int i = 0; i < input.length; i++) {
            input[i] = (float) Math.exp(input[i] - max);
            sum += input[i];
        }

        for (int i = 0; i < input.length; i++) {
            input[i] /= sum;
        }
    }

    // ----------------------------------------------------------------------
    // process current frame
    @Override
    public void analyze(@NonNull ImageProxy imageProxy) {
        if (imageProxy.getFormat() == PixelFormat.RGBA_8888){
            Bitmap bitmapImage = Bitmap.createBitmap(imageProxy.getWidth(),imageProxy.getHeight(),Bitmap.Config.ARGB_8888);
            bitmapImage.copyPixelsFromBuffer(imageProxy.getPlanes()[0].getBuffer());
            imageProxy.close();

            bitmapImage = rotateBitmap(bitmapImage, 90);

            // Map of labels and their corresponding probability
            TensorLabel labels = predictLabelStream(bitmapImage);

            String resultString = "";
            // Create a map to access the result based on label
            Map<String, Float> floatMap = labels.getMapWithFloatValue();
            resultString = getResultString(floatMap);
            classificationResults.setText(resultString);
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

    /*
    * Helper function for just getting the float value of the best result
    */
    public static float getBest(Map<String, Float> mapResults) {
        Map.Entry<String, Float> entryMax = null;
        for(Map.Entry<String, Float> entry: mapResults.entrySet()){
            if (entryMax == null || entry.getValue().compareTo(entryMax.getValue()) > 0) {
                entryMax = entry;
            }
        }
        return entryMax.getValue();
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
        String result = entryMax.getKey().trim();
        return result;
    }

} // class
