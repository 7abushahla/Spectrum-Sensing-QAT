#include <SDHCI.h>
#include <File.h>
#include <malloc.h> // For mallinfo()
#include <math.h>   // For sqrt()

// Import TensorFlow Lite
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "ParallelCNN_32_QAT_LTE_SNR_10_best_overall_model_INT8.h"  // Your model file after conversion

// Globals
namespace {
  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;
  int inference_count = 0;

  // TensorFlow Lite memory
  constexpr int kTensorArenaSize = (300 * 1024);
  uint8_t tensor_arena[kTensorArenaSize];
}

#define BAUDRATE 115200

SDClass SD;
File dataFile;

// Model input dimensions
const int NUM_SAMPLES = 128;
const int NUM_CHANNELS = 2;
const int NUM_CLASSES = 16;
const int NUM_ITERATIONS = 1000;

// Global buffer to store the fixed sample (read once from SD card)
float fixed_sample[NUM_SAMPLES * NUM_CHANNELS];

// Function to report memory usage
void ReportMemoryUsage(const char* context) {
  struct mallinfo mi = mallinfo(); // Get memory allocation info
  Serial.print(context);
  Serial.print(" Free memory: ");
  Serial.println(mi.fordblks, DEC); // Free space in the heap
}

void setupModel() {
  tflite::InitializeTarget();
  memset(tensor_arena, 0, kTensorArenaSize * sizeof(uint8_t));

  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  model = tflite::GetModel(model_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model version mismatch!");
    return;
  }

  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  ReportMemoryUsage("Before AllocateTensors(): ");
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors() failed");
    while (1);
  }
  ReportMemoryUsage("After AllocateTensors(): ");
  Serial.println("Area used bytes: " + String(interpreter->arena_used_bytes()));

  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("TensorFlow setup completed.\n");
}

// Function to load a fixed sample once from SD card
bool readFixedSampleFromSD(float* sample_buffer, int num_samples, int num_channels) {
  dataFile = SD.open("SDR/X_testLTE.csv");
  if (!dataFile) {
    Serial.println("Failed to open test data file.");
    return false;
  }

  int i = 0;
  while (dataFile.available() && i < num_samples * num_channels) {
    sample_buffer[i++] = dataFile.parseFloat();
  }
  dataFile.close();

  if (i != num_samples * num_channels) {
    Serial.println("Incomplete sample data read.");
    return false;
  }
  return true;
}

// Function to perform inference on the preloaded sample
void performInference(int num_samples, int num_channels, int num_classes, int num_iterations) {
  float inference_times[num_iterations]; // Store times for statistical analysis
  float total_time = 0;

  for (int i = 0; i < num_iterations; ++i) {
    // Copy the pre-loaded sample into the input tensor
    for (int j = 0; j < num_samples * num_channels; ++j) {
      input->data.f[j] = fixed_sample[j];
    }

    // Measure inference time
    unsigned long start_time = micros();
    if (interpreter->Invoke() != kTfLiteOk) {
      Serial.println("Inference failed.");
      return;
    }
    unsigned long end_time = micros();

    // Store inference time in milliseconds
    float inference_time = (end_time - start_time) / 1000.0;
    inference_times[i] = inference_time;
    total_time += inference_time;

    Serial.print("Iteration ");
    Serial.print(i + 1);
    Serial.print(" - Inference Time: ");
    Serial.print(inference_time, 4);
    Serial.println(" ms");
  }

  // Compute mean inference time
  float mean_time = total_time / num_iterations;

  // Compute standard deviation
  float sum_sq_diff = 0;
  for (int i = 0; i < num_iterations; ++i) {
    float diff = inference_times[i] - mean_time;
    sum_sq_diff += diff * diff;
  }
  float std_dev = (num_iterations > 1) ? sqrt(sum_sq_diff / (num_iterations - 1)) : 0;

  // Print final results
  Serial.println("\n=== Inference Time Statistics ===");
  Serial.print("Mean Inference Time: ");
  Serial.print(mean_time, 4);
  Serial.println(" ms");

  Serial.print("Standard Deviation: ");
  Serial.print(std_dev, 4);
  Serial.println(" ms");
}

void setup() {
  Serial.begin(BAUDRATE);
  while (!Serial) { ; }

  Serial.println("Initializing SD card...");
  while (!SD.begin()) {
    Serial.println("Insert SD card.");
  }
  Serial.println("SD card initialized.");

  // Load the fixed sample from SD card once
  if (!readFixedSampleFromSD(fixed_sample, NUM_SAMPLES, NUM_CHANNELS)) {
    Serial.println("Error reading fixed sample from SD. Halting.");
    while (1);
  }

  setupModel();
}

void loop() {
  Serial.println("Starting inference...");
  performInference(NUM_SAMPLES, NUM_CHANNELS, NUM_CLASSES, NUM_ITERATIONS);

  Serial.println("\nInference complete. Halting execution.");
  while (true); // Stop after reporting
}

