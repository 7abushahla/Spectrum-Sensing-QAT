#include <SDHCI.h>
#include <File.h>
#include <malloc.h> // For mallinfo()
#include <math.h>   // For sqrt()
#include <stdlib.h> // For atof()
#include <string.h> // For memcpy()

// Import TensorFlow stuff
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "ParallelCNN_32_QAT_SDR_best_overall_model_INT8.h"  // Your model file after conversion

// Globals, used for compatibility with Arduino-style sketches.
namespace {
  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;
  int inference_count = 0;

  // Memory for TensorFlow Lite
  constexpr int kTensorArenaSize = (100 * 1024);
  uint8_t tensor_arena[kTensorArenaSize];
}

#define BAUDRATE 115200

SDClass SD;
File dataFile;

// Define model dimensions as global constants
const int NUM_SAMPLES = 32;       // Number of rows per sample
const int NUM_CHANNELS = 2;       // Number of channels (I and Q)
const int NUM_CLASSES = 4;        // Number of output classes
const int NUM_ITERATIONS = 1500;  // Total inference iterations

// Global buffers
// fixed_sample will store the sample loaded from SD (and then normalized)
// raw_sample stores the original unnormalized data.
float fixed_sample[NUM_SAMPLES * NUM_CHANNELS];
float raw_sample[NUM_SAMPLES * NUM_CHANNELS];

/////////////////////////////////////////////
// Function to report memory usage
void ReportMemoryUsage(const char* context) {
  struct mallinfo mi = mallinfo();
  Serial.print(context);
  Serial.print(" Free memory: ");
  Serial.println(mi.fordblks, DEC);
}

/////////////////////////////////////////////
// Setup the TensorFlow Lite model and print its input dimensions.
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

  Serial.println("Model input:");
  Serial.print("Number of dimensions: ");
  Serial.println(input->dims->size);
  for (int n = 0; n < input->dims->size; ++n) {
    Serial.print("dims->data[");
    Serial.print(n);
    Serial.print("]: ");
    Serial.println(input->dims->data[n]);
  }
  Serial.print("Input type: ");
  Serial.println(input->type);

  Serial.println("\nModel output:");
  Serial.print("Number of dimensions: ");
  Serial.println(output->dims->size);
  for (int n = 0; n < output->dims->size; ++n) {
    Serial.print("dims->data[");
    Serial.print(n);
    Serial.print("]: ");
    Serial.println(output->dims->data[n]);
  }
  Serial.print("Output type: ");
  Serial.println(output->type);

  Serial.println("Completed TensorFlow setup");
  Serial.println();
}

/////////////////////////////////////////////
// Function to load a fixed sample from the CSV file on the SD card.
// The CSV file is formatted so that each sample is represented by two rows:
// first row: I values (32 columns)
// second row: Q values (32 columns)
bool readFixedSampleFromSD(float* sample_buffer, int num_samples, int num_channels) {
  dataFile = SD.open("SDR/X_test32_two_rows.csv");
  if (!dataFile) {
    Serial.println("Failed to open test data file.");
    return false;
  }
  String iLine = dataFile.readStringUntil('\n');
  String qLine = dataFile.readStringUntil('\n');
  dataFile.close();
  if (iLine.length() == 0 || qLine.length() == 0) {
    Serial.println("Incomplete sample data read.");
    return false;
  }
  float Ivalues[NUM_SAMPLES];
  float Qvalues[NUM_SAMPLES];
  int count = 0;
  char iLineBuf[iLine.length() + 1];
  iLine.toCharArray(iLineBuf, iLine.length() + 1);
  char* token = strtok(iLineBuf, ",");
  while (token != NULL && count < num_samples) {
    Ivalues[count++] = atof(token);
    token = strtok(NULL, ",");
  }
  if (count != num_samples) {
    Serial.println("Error: I row does not contain the expected number of samples.");
    return false;
  }
  count = 0;
  char qLineBuf[qLine.length() + 1];
  qLine.toCharArray(qLineBuf, qLine.length() + 1);
  token = strtok(qLineBuf, ",");
  while (token != NULL && count < num_samples) {
    Qvalues[count++] = atof(token);
    token = strtok(NULL, ",");
  }
  if (count != num_samples) {
    Serial.println("Error: Q row does not contain the expected number of samples.");
    return false;
  }
  for (int i = 0; i < num_samples; i++) {
    sample_buffer[i * num_channels + 0] = Ivalues[i];
    sample_buffer[i * num_channels + 1] = Qvalues[i];
  }
  Serial.print("Fixed sample dimensions: ");
  Serial.print(num_samples);
  Serial.print(" x ");
  Serial.println(num_channels);

  // for (int i = 0; i < num_samples; i++) {
  //   Serial.print("Row ");
  //   Serial.print(i);
  //   Serial.print(": I=");
  //   Serial.print(sample_buffer[i * num_channels + 0], 5);
  //   Serial.print(", Q=");
  //   Serial.println(sample_buffer[i * num_channels + 1], 5);
  // }
  return true;
}

/////////////////////////////////////////////
// Function to perform per-channel z-score normalization on a sample buffer.
// It computes and prints the computed (pre-normalization) channel means and standard deviations,
// then normalizes the sample so that each channel has mean 0 and std 1,
// and finally prints the post-normalization statistics.
void normalizeFixedSample(float* sample_buffer, int num_rows, int num_channels) {
  float computedMeans[NUM_CHANNELS];
  float computedStds[NUM_CHANNELS];
  for (int c = 0; c < num_channels; c++) {
    float sum = 0;
    for (int r = 0; r < num_rows; r++) {
      sum += sample_buffer[r * num_channels + c];
    }
    float mean = sum / num_rows;
    computedMeans[c] = mean;
    float sum_sq = 0;
    for (int r = 0; r < num_rows; r++) {
      float diff = sample_buffer[r * num_channels + c] - mean;
      sum_sq += diff * diff;
    }
    float std = sqrt(sum_sq / num_rows);
    if (fabs(std) < 1e-6) {
      std = 1.0;
      Serial.println("Warning: Standard deviation near zero for channel " + String(c) + ". Adjusting to 1.0.");
    }
    computedStds[c] = std;
  }

  // Serial.print("Computed channel means (float64): [");
  // for (int c = 0; c < num_channels; c++) {
  //   Serial.print(computedMeans[c], 8);
  //   if (c < num_channels - 1)
  //     Serial.print(" ");
  // }
  // Serial.println("]");
  // Serial.print("Computed channel standard deviations (float64): [");
  // for (int c = 0; c < num_channels; c++) {
  //   Serial.print(computedStds[c], 8);
  //   if (c < num_channels - 1)
  //     Serial.print(" ");
  // }
  // Serial.println("]");

  for (int c = 0; c < num_channels; c++) {
    for (int r = 0; r < num_rows; r++) {
      sample_buffer[r * num_channels + c] = (sample_buffer[r * num_channels + c] - computedMeans[c]) / computedStds[c];
    }
  }
  float newMeans[NUM_CHANNELS];
  float newStds[NUM_CHANNELS];
  for (int c = 0; c < num_channels; c++) {
    float newSum = 0;
    for (int r = 0; r < num_rows; r++) {
      newSum += sample_buffer[r * num_channels + c];
    }
    newMeans[c] = newSum / num_rows;
    float newSumSq = 0;
    for (int r = 0; r < num_rows; r++) {
      float val = sample_buffer[r * num_channels + c];
      newSumSq += val * val;
    }
    newStds[c] = sqrt(newSumSq / num_rows);
  }

  // Serial.print("Post-normalization channel means (float64): [");
  // for (int c = 0; c < num_channels; c++) {
  //   Serial.print(newMeans[c], 8);
  //   if (c < num_channels - 1)
  //     Serial.print(" ");
  // }
  // Serial.println("]");
  // Serial.print("Post-normalization channel standard deviations (float64): [");
  // for (int c = 0; c < num_channels; c++) {
  //   Serial.print(newStds[c], 8);
  //   if (c < num_channels - 1)
  //     Serial.print(" ");
  // }
  // Serial.println("]");
}

/////////////////////////////////////////////
// Function to perform inference using the already normalized fixed sample.
// The normalization is done once (outside this loop).
// We then run inference for NUM_ITERATIONS iterations and compute the average inference time.
void performInference(int num_samples, int num_channels, int num_classes, int num_iterations) {
  int batch = input->dims->data[0];    // Expected to be 1
  int rows = input->dims->data[1];       // Expected to be num_samples
  int channels = input->dims->data[2];   // Expected to be num_channels
  if (batch != 1 || rows != num_samples || channels != num_channels) {
    Serial.println("Error: Model input dimensions do not match expected fixed sample dimensions!");
    Serial.print("Model input dimensions: [");
    Serial.print(batch);
    Serial.print(", ");
    Serial.print(rows);
    Serial.print(", ");
    Serial.print(channels);
    Serial.println("]");
    return;
  } else {
    Serial.println("Model input dimensions verified: [1, " + String(rows) + ", " + String(channels) + "]");
  }
  
  float inference_times[NUM_ITERATIONS];
  float total_inf = 0;
  for (int i = 0; i < num_iterations; ++i) {
    for (int j = 0; j < num_samples * num_channels; ++j) {
      input->data.f[j] = fixed_sample[j];
    }
    unsigned long inf_start = micros();
    if (interpreter->Invoke() != kTfLiteOk) {
      Serial.println("Inference failed.");
      return;
    }
    unsigned long inf_end = micros();
    float inf_time = (inf_end - inf_start) / 1000.0;
    inference_times[i] = inf_time;
    total_inf += inf_time;
    Serial.print("Iteration ");
    Serial.print(i + 1);
    Serial.print(" - Inference Time: ");
    Serial.print(inf_time, 4);
    Serial.println(" ms");
  }
  
  float avg_inf = total_inf / num_iterations;
  float sum_sq_diff = 0;
  for (int i = 0; i < num_iterations; ++i) {
    float diff = inference_times[i] - avg_inf;
    sum_sq_diff += diff * diff;
  }
  float std_dev = (num_iterations > 1) ? sqrt(sum_sq_diff / (num_iterations - 1)) : 0;
  
  Serial.println("\n=== Timing Statistics ===");
  Serial.print("Average Inference Time: ");
  Serial.print(avg_inf, 4);
  Serial.println(" ms");
  Serial.print("Inference Time Standard Deviation: ");
  Serial.print(std_dev, 4);
  Serial.println(" ms");
}

/////////////////////////////////////////////
void setup() {
  Serial.begin(BAUDRATE);
  while (!Serial) { ; }
  Serial.println("Initializing SD card...");
  while (!SD.begin()) {
    Serial.println("Insert SD card.");
  }
  Serial.println("SD card initialized.");
  
  if (!readFixedSampleFromSD(fixed_sample, NUM_SAMPLES, NUM_CHANNELS)) {
    Serial.println("Error reading fixed sample from SD. Halting.");
    while (1);
  }
  memcpy(raw_sample, fixed_sample, sizeof(fixed_sample));
  
  // Normalize the fixed sample once and measure the time.
  unsigned long norm_start = micros();
  normalizeFixedSample(fixed_sample, NUM_SAMPLES, NUM_CHANNELS);
  unsigned long norm_end = micros();
  float norm_time = (norm_end - norm_start) / 1000.0; // in ms
  Serial.print("Normalization Time (one-time): ");
  Serial.print(norm_time, 4);
  Serial.println(" ms");
  
  setupModel();
}

/////////////////////////////////////////////
void loop() {
  Serial.println("Starting inference (using already normalized sample)...");
  performInference(NUM_SAMPLES, NUM_CHANNELS, NUM_CLASSES, NUM_ITERATIONS);
  
  // Compute and print end-to-end average time.
  // End-to-end time = one-time normalization time + average inference time.
  // (Since normalization is done only once.)
  int dummyIterations = 1; // normalization done once.
  // Assume we already printed norm_time in setup() and avg_inf in performInference.
  // For clarity, you could store avg_inf in a global variable.
  
  Serial.println("\nInference complete. Halting execution.");
  while (true);
}



//////////////////////////////////
// #include <SDHCI.h>
// #include <File.h>
// #include <malloc.h> // For mallinfo()
// #include <math.h>   // For sqrt()
// #include <stdlib.h> // For atof()
// #include <string.h> // For memcpy()

// // Import TensorFlow stuff
// #include "tensorflow/lite/micro/all_ops_resolver.h"
// #include "tensorflow/lite/micro/micro_error_reporter.h"
// #include "tensorflow/lite/micro/micro_interpreter.h"
// #include "tensorflow/lite/micro/system_setup.h"
// #include "tensorflow/lite/schema/schema_generated.h"
// #include "ParallelCNN_128_QAT_SDR_best_overall_model_INT8.h"  // Your model file after conversion

// // Globals, used for compatibility with Arduino-style sketches.
// namespace {
//   tflite::ErrorReporter* error_reporter = nullptr;
//   const tflite::Model* model = nullptr;
//   tflite::MicroInterpreter* interpreter = nullptr;
//   TfLiteTensor* input = nullptr;
//   TfLiteTensor* output = nullptr;
//   int inference_count = 0;

//   // Memory for TensorFlow Lite
//   constexpr int kTensorArenaSize = (100 * 1024);
//   uint8_t tensor_arena[kTensorArenaSize];
// }

// #define BAUDRATE 115200

// SDClass SD;
// File dataFile;

// // Define model dimensions as global constants
// const int NUM_SAMPLES = 128;       // Number of rows per sample
// const int NUM_CHANNELS = 2;       // Number of channels (I and Q)
// const int NUM_CLASSES = 4;        // Number of output classes
// const int NUM_ITERATIONS = 1000;  // Total inference iterations

// // Global buffers
// // fixed_sample will store the sample loaded from SD (it may later be normalized)
// // raw_sample will store the original unnormalized data.
// float fixed_sample[NUM_SAMPLES * NUM_CHANNELS];
// float raw_sample[NUM_SAMPLES * NUM_CHANNELS];

// /////////////////////////////////////////////
// // Function to report memory usage
// void ReportMemoryUsage(const char* context) {
//   struct mallinfo mi = mallinfo(); // Get memory allocation info
//   Serial.print(context);
//   Serial.print(" Free memory: ");
//   Serial.println(mi.fordblks, DEC); // Free space in the heap
// }

// /////////////////////////////////////////////
// // Setup the TensorFlow Lite model and print its input dimensions.
// void setupModel() {
//   tflite::InitializeTarget();
//   memset(tensor_arena, 0, kTensorArenaSize * sizeof(uint8_t));

//   static tflite::MicroErrorReporter micro_error_reporter;
//   error_reporter = &micro_error_reporter;

//   model = tflite::GetModel(model_tflite);
//   if (model->version() != TFLITE_SCHEMA_VERSION) {
//     Serial.println("Model version mismatch!");
//     return;
//   }

//   static tflite::AllOpsResolver resolver;
//   static tflite::MicroInterpreter static_interpreter(
//       model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
//   interpreter = &static_interpreter;

//   ReportMemoryUsage("Before AllocateTensors(): ");
//   if (interpreter->AllocateTensors() != kTfLiteOk) {
//     Serial.println("AllocateTensors() failed");
//     while (1);
//   }
//   ReportMemoryUsage("After AllocateTensors(): ");
//   Serial.println("Area used bytes: " + String(interpreter->arena_used_bytes()));

//   // Obtain pointers to the model's input and output tensors
//   input = interpreter->input(0);
//   output = interpreter->output(0);

//   // Print model input dimensions for verification.
//   Serial.println("Model input:");
//   Serial.print("Number of dimensions: ");
//   Serial.println(input->dims->size);
//   for (int n = 0; n < input->dims->size; ++n) {
//     Serial.print("dims->data[");
//     Serial.print(n);
//     Serial.print("]: ");
//     Serial.println(input->dims->data[n]);
//   }
//   Serial.print("Input type: ");
//   Serial.println(input->type);

//   Serial.println("\nModel output:");
//   Serial.print("Number of dimensions: ");
//   Serial.println(output->dims->size);
//   for (int n = 0; n < output->dims->size; ++n) {
//     Serial.print("dims->data[");
//     Serial.print(n);
//     Serial.print("]: ");
//     Serial.println(output->dims->data[n]);
//   }
//   Serial.print("Output type: ");
//   Serial.println(output->type);

//   Serial.println("Completed TensorFlow setup");
//   Serial.println();
// }

// /////////////////////////////////////////////
// // Function to load a fixed sample from the CSV file on the SD card.
// // The CSV file is formatted so that each sample is represented by two rows:
// // first row: I values (32 columns)
// // second row: Q values (32 columns)
// bool readFixedSampleFromSD(float* sample_buffer, int num_samples, int num_channels) {
//   dataFile = SD.open("SDR/X_test128_two_rows.csv");
//   if (!dataFile) {
//     Serial.println("Failed to open test data file.");
//     return false;
//   }

//   // Read the first two lines (first sample) from the CSV file.
//   String iLine = dataFile.readStringUntil('\n');
//   String qLine = dataFile.readStringUntil('\n');
//   dataFile.close();

//   if (iLine.length() == 0 || qLine.length() == 0) {
//     Serial.println("Incomplete sample data read.");
//     return false;
//   }

//   // Temporary arrays to hold I and Q channel values.
//   float Ivalues[NUM_SAMPLES];
//   float Qvalues[NUM_SAMPLES];

//   // Parse the I row.
//   int count = 0;
//   char iLineBuf[iLine.length() + 1];
//   iLine.toCharArray(iLineBuf, iLine.length() + 1);
//   char* token = strtok(iLineBuf, ",");
//   while (token != NULL && count < num_samples) {
//     Ivalues[count++] = atof(token);
//     token = strtok(NULL, ",");
//   }
//   if (count != num_samples) {
//     Serial.println("Error: I row does not contain the expected number of samples.");
//     return false;
//   }

//   // Parse the Q row.
//   count = 0;
//   char qLineBuf[qLine.length() + 1];
//   qLine.toCharArray(qLineBuf, qLine.length() + 1);
//   token = strtok(qLineBuf, ",");
//   while (token != NULL && count < num_samples) {
//     Qvalues[count++] = atof(token);
//     token = strtok(NULL, ",");
//   }
//   if (count != num_samples) {
//     Serial.println("Error: Q row does not contain the expected number of samples.");
//     return false;
//   }

//   // Combine I and Q values into the sample buffer in row-major order.
//   // Each row represents one sample point with two channels: channel 0 is I, channel 1 is Q.
//   for (int i = 0; i < num_samples; i++) {
//     sample_buffer[i * num_channels + 0] = Ivalues[i];
//     sample_buffer[i * num_channels + 1] = Qvalues[i];
//   }

//   // Print the sample dimensions.
//   Serial.print("Fixed sample dimensions: ");
//   Serial.print(num_samples);
//   Serial.print(" x ");
//   Serial.println(num_channels);

//   // // Print the sample values with 5 decimal places.
//   // for (int i = 0; i < num_samples; i++) {
//   //   Serial.print("Row ");
//   //   Serial.print(i);
//   //   Serial.print(": I=");
//   //   Serial.print(sample_buffer[i * num_channels + 0], 5);
//   //   Serial.print(", Q=");
//   //   Serial.println(sample_buffer[i * num_channels + 1], 5);
//   // }

//   return true;
// }

// /////////////////////////////////////////////
// // Function to perform per-channel z-score normalization.
// // For each channel, it computes the mean and standard deviation over the sample,
// // prints the computed means and standard deviations (pre-normalization),
// // then normalizes that channel so its values have mean 0 and std 1.
// // After normalization, it computes and prints the new means and standard deviations (post-normalization)
// // and finally prints the normalized sample.
// void normalizeFixedSample(float* sample_buffer, int num_rows, int num_channels) {
//   float computedMeans[NUM_CHANNELS];
//   float computedStds[NUM_CHANNELS];

//   // Compute mean and standard deviation per channel (pre-normalization).
//   for (int c = 0; c < num_channels; c++) {
//     float sum = 0;
//     for (int r = 0; r < num_rows; r++) {
//       sum += sample_buffer[r * num_channels + c];
//     }
//     float mean = sum / num_rows;
//     computedMeans[c] = mean;

//     float sum_sq = 0;
//     for (int r = 0; r < num_rows; r++) {
//       float diff = sample_buffer[r * num_channels + c] - mean;
//       sum_sq += diff * diff;
//     }
//     float std = sqrt(sum_sq / num_rows);
//     if (fabs(std) < 1e-6) {
//       std = 1.0;  // Avoid division by zero.
//       Serial.println("Warning: Standard deviation near zero for channel " + String(c) + ". Adjusting to 1.0.");
//     }
//     computedStds[c] = std;
//   }

//   // // Print the computed means and standard deviations (pre-normalization).
//   // Serial.print("Computed channel means (float64): [");
//   // for (int c = 0; c < num_channels; c++) {
//   //   Serial.print(computedMeans[c], 8);
//   //   if(c < num_channels - 1)
//   //     Serial.print(" ");
//   // }
//   // Serial.println("]");
  
//   // Serial.print("Computed channel standard deviations (float64): [");
//   // for (int c = 0; c < num_channels; c++) {
//   //   Serial.print(computedStds[c], 8);
//   //   if(c < num_channels - 1)
//   //     Serial.print(" ");
//   // }
//   // Serial.println("]");

//   // Normalize the sample buffer per channel.
//   for (int c = 0; c < num_channels; c++) {
//     for (int r = 0; r < num_rows; r++) {
//       sample_buffer[r * num_channels + c] = (sample_buffer[r * num_channels + c] - computedMeans[c]) / computedStds[c];
//     }
//   }

//   // Compute new statistics (post-normalization).
//   float newMeans[NUM_CHANNELS];
//   float newStds[NUM_CHANNELS];
//   for (int c = 0; c < num_channels; c++) {
//     float newSum = 0;
//     for (int r = 0; r < num_rows; r++) {
//       newSum += sample_buffer[r * num_channels + c];
//     }
//     newMeans[c] = newSum / num_rows;
    
//     float newSumSq = 0;
//     for (int r = 0; r < num_rows; r++) {
//       float val = sample_buffer[r * num_channels + c];
//       newSumSq += val * val;
//     }
//     newStds[c] = sqrt(newSumSq / num_rows);
//   }
  
//   // // Print the post-normalization means and standard deviations.
//   // Serial.print("Post-normalization channel means (float64): [");
//   // for (int c = 0; c < num_channels; c++) {
//   //   Serial.print(newMeans[c], 8);
//   //   if(c < num_channels - 1)
//   //     Serial.print(" ");
//   // }
//   // Serial.println("]");
  
//   // Serial.print("Post-normalization channel standard deviations (float64): [");
//   // for (int c = 0; c < num_channels; c++) {
//   //   Serial.print(newStds[c], 8);
//   //   if(c < num_channels - 1)
//   //     Serial.print(" ");
//   // }
//   // Serial.println("]");

//   // Print the normalized sample.
//   // Serial.println("Normalized fixed sample:");
//   // for (int r = 0; r < num_rows; r++) {
//   //   Serial.print("Row ");
//   //   Serial.print(r);
//   //   Serial.print(": ");
//   //   for (int c = 0; c < num_channels; c++) {
//   //     Serial.print(sample_buffer[r * num_channels + c], 8);
//   //     Serial.print(" ");
//   //   }
//   //   Serial.println();
//   // }
// }

// /////////////////////////////////////////////
// // Function to perform inference using the fixed sample.
// // In each iteration, we first copy the raw (unnormalized) sample into a temporary buffer,
// // measure the time to normalize it, then perform inference using the normalized data.
// // Finally, we print the average normalization time, average inference time,
// // and the average total end-to-end time (normalization + inference).
// void performInference(int num_samples, int num_channels, int num_classes, int num_iterations) {
//   // Verify that the model input dimensions match.
//   int batch = input->dims->data[0];    // Expected to be 1
//   int rows = input->dims->data[1];       // Expected to be num_samples
//   int channels = input->dims->data[2];   // Expected to be num_channels
//   if (batch != 1 || rows != num_samples || channels != num_channels) {
//     Serial.println("Error: Model input dimensions do not match expected fixed sample dimensions!");
//     Serial.print("Model input dimensions: [");
//     Serial.print(batch);
//     Serial.print(", ");
//     Serial.print(rows);
//     Serial.print(", ");
//     Serial.print(channels);
//     Serial.println("]");
//     return;
//   } else {
//     Serial.println("Model input dimensions verified: [1, " + String(rows) + ", " + String(channels) + "]");
//   }

//   float inference_times[NUM_ITERATIONS]; // Store inference times (ms)
//   float norm_times[NUM_ITERATIONS];      // Store normalization times (ms)
//   float total_times[NUM_ITERATIONS];     // Store total times (norm + inference) per iteration
//   float total_norm = 0;
//   float total_inf = 0;
//   float total_end = 0;

//   // Temporary buffer for normalized sample.
//   float norm_sample[NUM_SAMPLES * NUM_CHANNELS];

//   for (int i = 0; i < num_iterations; ++i) {
//     // Copy the raw sample into norm_sample.
//     for (int j = 0; j < num_samples * num_channels; ++j) {
//       norm_sample[j] = raw_sample[j];
//     }
//     // Measure normalization time.
//     unsigned long norm_start = micros();
//     normalizeFixedSample(norm_sample, num_samples, num_channels);
//     unsigned long norm_end = micros();
//     float norm_time = (norm_end - norm_start) / 1000.0; // in ms
//     norm_times[i] = norm_time;
//     total_norm += norm_time;

//     // Copy normalized data into the model's input tensor.
//     for (int j = 0; j < num_samples * num_channels; ++j) {
//       input->data.f[j] = norm_sample[j];
//     }
    
//     // Measure inference time.
//     unsigned long inf_start = micros();
//     if (interpreter->Invoke() != kTfLiteOk) {
//       Serial.println("Inference failed.");
//       return;
//     }
//     unsigned long inf_end = micros();
//     float inf_time = (inf_end - inf_start) / 1000.0; // in ms
//     inference_times[i] = inf_time;
//     total_inf += inf_time;

//     float end_to_end = norm_time + inf_time;
//     total_end += end_to_end;
//     total_times[i] = end_to_end;

//     Serial.print("Iteration ");
//     Serial.print(i + 1);
//     Serial.print(" - Normalization Time: ");
//     Serial.print(norm_time, 4);
//     Serial.print(" ms, Inference Time: ");
//     Serial.print(inf_time, 4);
//     Serial.print(" ms, Total: ");
//     Serial.print(end_to_end, 4);
//     Serial.println(" ms");
//   }

//   float avg_norm = total_norm / num_iterations;
//   float avg_inf = total_inf / num_iterations;
//   float avg_end = total_end / num_iterations;

//   // Compute standard deviation for inference times (optional)
//   float sum_sq_diff = 0;
//   for (int i = 0; i < num_iterations; ++i) {
//     float diff = inference_times[i] - avg_inf;
//     sum_sq_diff += diff * diff;
//   }
//   float std_dev = (num_iterations > 1) ? sqrt(sum_sq_diff / (num_iterations - 1)) : 0;

//   Serial.println("\n=== Timing Statistics ===");
//   Serial.print("Average Normalization Time: ");
//   Serial.print(avg_norm, 4);
//   Serial.println(" ms");
//   Serial.print("Average Inference Time: ");
//   Serial.print(avg_inf, 4);
//   Serial.println(" ms");
//   Serial.print("Average End-to-End Time: ");
//   Serial.print(avg_end, 4);
//   Serial.println(" ms");
//   Serial.print("Inference Time Standard Deviation: ");
//   Serial.print(std_dev, 4);
//   Serial.println(" ms");
// }

// /////////////////////////////////////////////
// void setup() {
//   Serial.begin(BAUDRATE);
//   while (!Serial) { ; }

//   Serial.println("Initializing SD card...");
//   while (!SD.begin()) {
//     Serial.println("Insert SD card.");
//   }
//   Serial.println("SD card initialized.");

//   // Load the fixed sample from the CSV file.
//   if (!readFixedSampleFromSD(fixed_sample, NUM_SAMPLES, NUM_CHANNELS)) {
//     Serial.println("Error reading fixed sample from SD. Halting.");
//     while (1);
//   }

//   // Save a copy of the raw (unnormalized) sample.
//   memcpy(raw_sample, fixed_sample, sizeof(fixed_sample));

//   setupModel();
// }

// /////////////////////////////////////////////
// void loop() {
//   Serial.println("Starting inference with normalization...");
//   performInference(NUM_SAMPLES, NUM_CHANNELS, NUM_CLASSES, NUM_ITERATIONS);

//   Serial.println("\nInference complete. Halting execution.");
//   while (true); // Stop after reporting.
// }

// //////////////////////////



