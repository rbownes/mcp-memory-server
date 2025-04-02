use anyhow::{Result};
use async_trait::async_trait;
use std::path::{Path, PathBuf};
use std::sync::Arc;

// ONNX Runtime related imports
// *** FIX: Removed Allocator import ***
use ort::{Environment, Session, SessionBuilder, Value, GraphOptimizationLevel, LoggingLevel, OrtError};
use ort::tensor::OrtOwnedTensor;
use tokenizers::Tokenizer;
// Import ndarray types needed
// *** FIX: Removed unused Ix3, CowRepr, Dim ***
use ndarray::{Array, ArrayBase, Axis, Ix2, IxDyn, OwnedRepr, Data, ArrayView}; // Keep needed types

#[derive(thiserror::Error, Debug)]
pub enum EmbeddingError {
    #[error("Model file not found: {0}")]
    ModelNotFound(String),
    #[error("Tokenizer file not found: {0}")]
    TokenizerNotFound(String),
    #[error("Model loading failed: {0}")]
    ModelLoadError(String),
    #[error("Tokenizer loading failed: {0}")]
    TokenizerLoadError(String),
    #[error("Embedding generation failed: {0}")]
    InferenceError(String),
    #[error("Input tokenization failed: {0}")]
    TokenizationError(String),
    #[error("Tensor creation failed: {0}")]
    TensorError(String),
    #[error("Output processing failed: {0}")]
    OutputProcessingError(String),
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

// Convert ort::OrtError to EmbeddingError
impl From<OrtError> for EmbeddingError {
    fn from(err: OrtError) -> Self {
        EmbeddingError::ModelLoadError(err.to_string())
    }
}

// Convert tokenizers::Error to EmbeddingError
impl From<tokenizers::Error> for EmbeddingError {
     fn from(err: tokenizers::Error) -> Self {
         EmbeddingError::TokenizationError(err.to_string())
     }
}


#[async_trait]
pub trait EmbeddingGenerator: Send + Sync {
    async fn generate_embedding(&self, text: &str) -> Result<Vec<f32>, EmbeddingError>;
    fn get_embedding_size(&self) -> usize;
    fn name(&self) -> &'static str;
}

// --- Dummy Embedding Generator (Unchanged) ---
pub struct DummyEmbeddingGenerator {
    embedding_size: usize,
}
// ... (implementation unchanged) ...
impl DummyEmbeddingGenerator {
    pub fn new(embedding_size: usize) -> Self {
        Self { embedding_size }
    }
}
#[async_trait]
impl EmbeddingGenerator for DummyEmbeddingGenerator {
    async fn generate_embedding(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        let text_len = text.len() as f32;
        let mut embedding = Vec::with_capacity(self.embedding_size);
        for i in 0..self.embedding_size {
            let value = ((i as f32 * 0.1) + (text_len * 0.01)).sin() * 0.5;
            embedding.push(value);
        }
        let magnitude: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if magnitude > 0.0 {
            for value in &mut embedding {
                *value /= magnitude;
            }
        }
        Ok(embedding)
    }
    fn get_embedding_size(&self) -> usize { self.embedding_size }
    fn name(&self) -> &'static str { "Dummy" }
}


// --- Real ONNX Embedding Generator Implementation ---
pub struct OnnxEmbeddingGenerator {
    session: Session,
    tokenizer: Tokenizer,
    embedding_size: usize,
}

impl OnnxEmbeddingGenerator {
    pub fn new(model_path: PathBuf, tokenizer_path: Option<PathBuf>, embedding_size: usize) -> Result<Self, EmbeddingError> {
        tracing::info!("Initializing ONNX embedding generator...");
        // ... (rest of the new function is unchanged) ...
        tracing::info!("Model path: {:?}", model_path);

        if !model_path.exists() {
            return Err(EmbeddingError::ModelNotFound(format!("Model file not found: {:?}", model_path)));
        }

        let actual_tokenizer_path = tokenizer_path.unwrap_or_else(|| {
            let parent_dir = model_path.parent().unwrap_or_else(|| Path::new("."));
            parent_dir.join("tokenizer.json")
        });
        tracing::info!("Tokenizer path: {:?}", actual_tokenizer_path);

        if !actual_tokenizer_path.exists() {
            return Err(EmbeddingError::TokenizerNotFound(format!("Tokenizer file not found: {:?}", actual_tokenizer_path)));
        }

        let environment = Arc::new(Environment::builder()
            .with_name("mcp-memory-onnx")
            .with_log_level(LoggingLevel::Warning)
            .build()?);

        let session = SessionBuilder::new(&environment)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?
            .with_model_from_file(model_path)
            .map_err(|e| EmbeddingError::ModelLoadError(e.to_string()))?;
        tracing::info!("ONNX session created successfully.");

        let tokenizer = Tokenizer::from_file(&actual_tokenizer_path)
            .map_err(|e| EmbeddingError::TokenizerLoadError(format!("Failed to load tokenizer: {}", e)))?;
        tracing::info!("Tokenizer loaded successfully.");

        tracing::info!("Using configured embedding size: {}", embedding_size);

        Ok(Self {
            session,
            tokenizer,
            embedding_size,
        })
    }

    // Accepts generic ArrayBase with dynamic dimensions (IxDyn)
    fn mean_pooling<S: Data<Elem = f32>>(
        last_hidden_state: &ArrayBase<S, IxDyn>, // Accept generic ArrayBase with IxDyn
        attention_mask: &Array<i64, Ix2>
    ) -> Result<Array<f32, Ix2>, EmbeddingError> {

        // Check dimensions
        if last_hidden_state.ndim() != 3 {
            return Err(EmbeddingError::OutputProcessingError(format!(
                "Expected 3 dimensions for last_hidden_state, got {}", last_hidden_state.ndim()
            )));
        }
        let batch_size = last_hidden_state.shape()[0];
        let sequence_length = last_hidden_state.shape()[1];
        let hidden_size = last_hidden_state.shape()[2];

        if attention_mask.shape()[0] != batch_size || attention_mask.shape()[1] != sequence_length {
             return Err(EmbeddingError::OutputProcessingError(format!(
                 "Attention mask shape {:?} incompatible with hidden state shape {:?}",
                 attention_mask.shape(), last_hidden_state.shape()
             )));
        }

        // Reshape attention mask for broadcasting
        let mapped_mask = attention_mask.mapv(|x| x as f32);
        let inserted_axis_mask = mapped_mask.insert_axis(Axis(2));
        let expanded_attention_mask = inserted_axis_mask
            .broadcast((batch_size, sequence_length, hidden_size))
            .ok_or_else(|| EmbeddingError::OutputProcessingError("Failed to broadcast attention mask".to_string()))?;

        // *** FIX E0599: Convert inputs to owned arrays for multiplication ***
        let owned_expanded_mask = expanded_attention_mask.to_owned();
        let owned_last_hidden_state = last_hidden_state.to_owned(); // Convert generic S to owned

        // Perform multiplication with owned arrays
        let masked_hidden_state = &owned_last_hidden_state * &owned_expanded_mask;


        let sum_embeddings = masked_hidden_state.sum_axis(Axis(1));
        let sum_mask = owned_expanded_mask.sum_axis(Axis(1));
        let clamped_sum_mask = sum_mask.mapv(|x| x.max(1e-9));
        let mean_pooled_embeddings = sum_embeddings / clamped_sum_mask;

        Ok(mean_pooled_embeddings)
    }

    fn normalize_l2(v: &mut [f32]) {
        let norm = (v.iter().map(|&x| x * x).sum::<f32>()).sqrt();
        if norm > 1e-9 {
            for x in v.iter_mut() {
                *x /= norm;
            }
        }
    }
}

#[async_trait]
impl EmbeddingGenerator for OnnxEmbeddingGenerator {
    async fn generate_embedding(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        let inputs = vec![text];
        let encodings = self.tokenizer.encode_batch(inputs, true)?;

        if encodings.is_empty() {
            return Err(EmbeddingError::TokenizationError("Tokenizer produced no encodings.".to_string()));
        }
        let encoding = &encodings[0];

        let ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
        let mask: Vec<i64> = encoding.get_attention_mask().iter().map(|&x| x as i64).collect();
        let type_ids: Vec<i64> = encoding.get_type_ids().iter().map(|&x| x as i64).collect();

        let sequence_length = ids.len();
        let batch_size = 1;

        let allocator = self.session.allocator();

        // Define the shape for the input tensors
        let input_shape: Vec<i64> = vec![batch_size as i64, sequence_length as i64];

        // *** FIX E0308: Create tensors directly and copy data ***

        // Create tensor for input_ids
        let mut ids_tensor = Value::create_tensor::<i64>(allocator, &input_shape)?;
        // Ensure the ndarray is in standard layout for slicing
        let ids_array = Array::from_shape_vec((batch_size, sequence_length), ids)
            .map_err(|e| EmbeddingError::TensorError(format!("Failed to create ids ndarray: {}", e)))?
            .as_standard_layout()
            .to_owned(); // Use owned standard layout
        ids_tensor.tensor_data_mut()?
            .copy_from_slice(ids_array.as_slice().ok_or_else(|| EmbeddingError::TensorError("Failed to get slice from ids_array".to_string()))?);

        // Create tensor for attention_mask
        let mut mask_tensor = Value::create_tensor::<i64>(allocator, &input_shape)?;
        let mask_array = Array::from_shape_vec((batch_size, sequence_length), mask.clone()) // Clone mask for pooling later
             .map_err(|e| EmbeddingError::TensorError(format!("Failed to create mask ndarray: {}", e)))?
             .as_standard_layout()
             .to_owned();
        mask_tensor.tensor_data_mut()?
            .copy_from_slice(mask_array.as_slice().ok_or_else(|| EmbeddingError::TensorError("Failed to get slice from mask_array".to_string()))?);

        // Create tensor for token_type_ids
        let mut type_ids_tensor = Value::create_tensor::<i64>(allocator, &input_shape)?;
        let type_ids_array = Array::from_shape_vec((batch_size, sequence_length), type_ids)
             .map_err(|e| EmbeddingError::TensorError(format!("Failed to create type_ids ndarray: {}", e)))?
             .as_standard_layout()
             .to_owned();
        type_ids_tensor.tensor_data_mut()?
             .copy_from_slice(type_ids_array.as_slice().ok_or_else(|| EmbeddingError::TensorError("Failed to get slice from type_ids_array".to_string()))?);


        // Collect tensors for input
        let inputs_onnx = vec![ids_tensor, mask_tensor, type_ids_tensor];


        // Run inference
        let outputs = self.session.run(inputs_onnx)?;

        // Extract output tensor
        let output_tensor: OrtOwnedTensor<f32, IxDyn> = outputs[0].try_extract()?;

        // *** FIX E0308: Use inferred type for the view ***
        let last_hidden_state_view = output_tensor.view(); // Type is inferred

        // Call mean_pooling with the view
        // Pass the mask_array (Ix2) created earlier
        let pooled_embedding_array = Self::mean_pooling(&last_hidden_state_view, &mask_array)?;

        // Extract the final embedding vector
        let mut embedding: Vec<f32> = pooled_embedding_array
            .row(0)
            .to_vec();

        // Optional: Verify size
        if embedding.len() != self.embedding_size {
             tracing::warn!(
                 "Actual model output size ({}) differs from configured embedding size ({}). Using actual size.",
                 embedding.len(), self.embedding_size
             );
        }

        // Normalize
        Self::normalize_l2(&mut embedding);

        Ok(embedding)
    }

     fn get_embedding_size(&self) -> usize {
        self.embedding_size
    }

    fn name(&self) -> &'static str {
        "ONNX"
    }
}
