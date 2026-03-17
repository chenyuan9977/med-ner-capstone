# MEDICAL NER & CONTRASTIVE RETRIEVAL SYSTEM - GRADIO GUI
# 
# CS687 Capstone Project - Chen Yuan
# Two-Stage Medical Named Entity Recognition with Contrastive Learning
#

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
import numpy as np
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModel, pipeline
import faiss


# CONFIGURATION
class Config:
    # Update these paths to match your local setup
    STAGE1_MODEL_PATH = "./models/stage1_ner"
    STAGE2_MODEL_PATH = "./models/stage2_final"
    
    # Model settings
    MAX_LENGTH = 128
    TOP_K = 5
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"🖥️ Using device: {Config.DEVICE}")


# LOAD MODELS
print("Loading models... (this may take a moment)")

# Stage 1: NER Model
try:
    ner_pipeline = pipeline(
        "ner",
        model=Config.STAGE1_MODEL_PATH,
        tokenizer=Config.STAGE1_MODEL_PATH,
        aggregation_strategy="simple",
        device=0 if torch.cuda.is_available() else -1
    )
    print("✅ Stage 1 NER model loaded")
except Exception as e:
    print(f"⚠️ Could not load Stage 1 model: {e}")
    print("   Using mock NER for demo purposes")
    ner_pipeline = None

# Stage 2: Contrastive Model
try:
    # Load tokenizer
    stage2_tokenizer = AutoTokenizer.from_pretrained(f"{Config.STAGE2_MODEL_PATH}/tokenizer")
    
    # Load model
    with open(f"{Config.STAGE2_MODEL_PATH}/config.json", 'r') as f:
        stage2_config = json.load(f)
    
    stage2_model = AutoModel.from_pretrained(stage2_config['model_name'])
    stage2_model.resize_token_embeddings(len(stage2_tokenizer))
    stage2_model.load_state_dict(torch.load(f"{Config.STAGE2_MODEL_PATH}/model.pt", map_location=Config.DEVICE))
    stage2_model.to(Config.DEVICE)
    stage2_model.eval()
    
    # Load candidate bank
    with open(f"{Config.STAGE2_MODEL_PATH}/candidate_bank.json", 'r') as f:
        candidate_bank = json.load(f)
    
    # Load FAISS index
    faiss_index = faiss.read_index(f"{Config.STAGE2_MODEL_PATH}/faiss_index.bin")
    
    print("✅ Stage 2 Contrastive model loaded")
    print(f"   Candidates: {len(candidate_bank)}")
    
except Exception as e:
    print(f"⚠️ Could not load Stage 2 model: {e}")
    print("   Using mock retrieval for demo purposes")
    stage2_model = None
    stage2_tokenizer = None
    candidate_bank = []
    faiss_index = None

print("✅ All models loaded!")


# CORE PREDICTION FUNCTIONS
def extract_entities_stage1(text):
    """
    Stage 1: Extract medical entities using NER.
    Returns list of (entity_text, label, start, end, confidence)
    """
    if ner_pipeline is None:
        # Mock response for demo
        return [
            {"word": "diabetes", "entity_group": "Disease", "score": 0.95, "start": 10, "end": 18},
            {"word": "insulin", "entity_group": "Medication", "score": 0.92, "start": 35, "end": 42},
        ]
    
    entities = ner_pipeline(text)
    return entities


def normalize_entity_stage2(text, entity_text):
    """
    Stage 2: Normalize an entity using contrastive retrieval.
    Returns list of (candidate, score) tuples.
    """
    if stage2_model is None or faiss_index is None:
        # Mock response for demo
        return [
            ("diabetes mellitus", 0.89),
            ("type 2 diabetes", 0.76),
            ("diabetic condition", 0.71),
        ]
    
    # Create marked query
    marked_query = text.replace(entity_text, f"[E]{entity_text}[/E]", 1)
    
    # Encode
    encoded = stage2_tokenizer(
        marked_query,
        padding=True,
        truncation=True,
        max_length=Config.MAX_LENGTH,
        return_tensors='pt'
    )
    
    # Get embedding
    with torch.no_grad():
        outputs = stage2_model(
            input_ids=encoded['input_ids'].to(Config.DEVICE),
            attention_mask=encoded['attention_mask'].to(Config.DEVICE)
        )
        query_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    
    # Normalize
    query_emb = query_emb / np.linalg.norm(query_emb)
    query_emb = query_emb.astype(np.float32)
    
    # Search
    scores, indices = faiss_index.search(query_emb, Config.TOP_K)
    
    results = []
    for idx, score in zip(indices[0], scores[0]):
        results.append((candidate_bank[idx], float(score)))
    
    return results


def predict_medical_entities(text):
    """
    Main prediction function: Two-stage NER + Normalization.
    
    Returns:
        highlighted_text: List of (text, label) tuples for gr.HighlightedText
        entity_table: List of dicts for gr.Dataframe
        json_output: Detailed JSON output
    """
    if not text.strip():
        return [], [], {}
    
    # Stage 1: NER
    entities = extract_entities_stage1(text)
    
    # Prepare highlighted text
    # Sort entities by start position (reverse) for proper text reconstruction
    sorted_entities = sorted(entities, key=lambda x: x.get('start', 0))
    
    highlighted_parts = []
    last_end = 0
    
    for ent in sorted_entities:
        start = ent.get('start', 0)
        end = ent.get('end', len(ent['word']))
        
        # Add text before entity
        if start > last_end:
            highlighted_parts.append((text[last_end:start], None))
        
        # Add entity with label
        entity_text = ent['word']
        label = ent['entity_group']
        highlighted_parts.append((entity_text, label))
        
        last_end = end
    
    # Add remaining text
    if last_end < len(text):
        highlighted_parts.append((text[last_end:], None))
    
    # Stage 2: Normalization for each entity
    entity_table = []
    json_results = {
        "input_text": text,
        "entities": []
    }
    
    for ent in entities:
        entity_text = ent['word']
        label = ent['entity_group']
        confidence = ent['score']
        
        # Get normalized forms
        normalized = normalize_entity_stage2(text, entity_text)
        
        # Best match
        best_match = normalized[0][0] if normalized else entity_text
        best_score = normalized[0][1] if normalized else 0.0
        
        # For table
        entity_table.append({
            "Original Entity": entity_text,
            "Type": label,
            "NER Confidence": f"{confidence:.2%}",
            "Normalized Form": best_match,
            "Match Score": f"{best_score:.3f}",
            "Alternatives": ", ".join([n[0] for n in normalized[1:3]])
        })
        
        # For JSON
        json_results["entities"].append({
            "original": entity_text,
            "type": label,
            "ner_confidence": float(confidence),
            "normalized": best_match,
            "match_score": float(best_score),
            "all_matches": [{"term": n[0], "score": float(n[1])} for n in normalized]
        })
    
    return highlighted_parts, entity_table, json_results


# COLOR SCHEME FOR ENTITY TYPES

ENTITY_COLORS = {
    "Disease": "#FF6B6B",      # Red
    "DISEASE": "#FF6B6B",
    "Medication": "#4ECDC4",    # Teal
    "MEDICATION": "#4ECDC4",
    "Symptom": "#FFE66D",       # Yellow
    "SYMPTOM": "#FFE66D",
    "Treatment": "#95E1D3",     # Mint
    "TREATMENT": "#95E1D3",
    "Test": "#DDA0DD",          # Plum
    "TEST": "#DDA0DD",
    "Anatomy": "#98D8C8",       # Seafoam
    "ANATOMY": "#98D8C8",
}

# GRADIO INTERFACE

# Custom CSS for professional styling
custom_css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #1a5276 0%, #2874a6 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    
    .header h1 {
        margin: 0;
        font-size: 2em;
    }
    
    .header p {
        margin: 10px 0 0 0;
        opacity: 0.9;
    }
    
    .stage-header {
        background: #e9ecef;
        padding: 10px 15px;
        border-left: 4px solid #1a5276;
        margin: 15px 0;
        font-weight: bold;
        font-size: 30px;
        color: #1a5276;
    }
    
    .info-box {
        background: #e3f2fd;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        color: #0d47a1;
        border: 1px solid #bbdefb;
    }
    
    .footer {
        text-align: center;
        margin-top: 30px;
        padding: 20px;
        background: #f8f9fa;
        border-radius: 8px;
        color: #444;
        border: 1px solid #e9ecef;
    }
    
    .legend-item {
        display: inline-block;
        margin: 5px 10px;
        padding: 5px 10px;
        border-radius: 4px;
        font-size: 0.9em;
    }
"""

# Sample medical texts for examples
EXAMPLE_TEXTS = [
    "The patient was diagnosed with type 2 diabetes mellitus and hypertension. Current medications include metformin 500mg twice daily and lisinopril 10mg once daily.",
    
    "Chief complaint: chest pain and shortness of breath. History reveals previous myocardial infarction. Patient reports occasional palpitations and dizziness.",
    
    "Assessment: Patient presents with symptoms consistent with influenza including fever, cough, and myalgia. Recommend rest and acetaminophen for symptom management.",
    
    "The MRI revealed a herniated disc at L4-L5 causing sciatica. Patient has been experiencing chronic lower back pain and numbness in the left leg.",
    
    "Diagnosed with stage 2 breast cancer. Treatment plan includes chemotherapy followed by radiation therapy. Patient also has a history of osteoporosis.",
]


def create_interface():
    """Create the Gradio interface."""
    
    with gr.Blocks(title="Medical NER System") as demo:
        
        # Header
        gr.HTML("""
            <div class="header">
                <h1> Medical NER & Contrastive Retrieval System</h1>
                <p>Two-Stage Named Entity Recognition with Entity Normalization</p>
                <p style="font-size: 0.9em; opacity: 0.8;">CS687 Capstone Project</p>
            </div>
        """)
        
        # Main layout
        with gr.Row():
            # Left column: Input
            with gr.Column(scale=1):
                gr.HTML('<div class="stage-header"> Input Clinical Text</div>')
                
                input_text = gr.Textbox(
                    label="",
                    placeholder="Paste clinical notes or medical text here...\n\nExample: The patient was diagnosed with diabetes and prescribed metformin.",
                    lines=8,
                    max_lines=15
                )
                
                with gr.Row():
                    clear_btn = gr.Button("🗑️ Clear", variant="secondary")
                    submit_btn = gr.Button("🔍 Analyze", variant="primary")
                
                # Info box
                # gr.HTML("""
                #     <div class="info-box">
                #         <strong>How it works:</strong><br>
                #         <b>Stage 1 (NER):</b> BioBERT identifies medical entities<br>
                #         <b>Stage 2 (Normalization):</b> Contrastive learning maps entities to standard medical terms
                #     </div>
                # """)
        
        # Results section
        gr.HTML('<div class="stage-header">Stage 1: Named Entity Recognition</div>')
        
        # Legend
        gr.HTML("""
            <div style="margin: 10px 0;">
                <span class="legend-item" style="background: #FF6B6B; color: white;">Disease</span>
                <!-- <span class="legend-item" style="background: #4ECDC4; color: white;">Medication</span>
                <span class="legend-item" style="background: #FFE66D; color: black;">Symptom</span>
                <span class="legend-item" style="background: #95E1D3; color: black;">Treatment</span>
                <span class="legend-item" style="background: #DDA0DD; color: black;">Test</span> -->
            </div>
        """)
        
        highlighted_output = gr.HighlightedText(
            label="Detected Entities",
            color_map=ENTITY_COLORS,
            show_legend=False
        )
        
        gr.HTML('<div class="stage-header">Stage 2: Entity Normalization (Contrastive Retrieval)</div>')
        
        with gr.Row():
            with gr.Column(scale=2):
                entity_table = gr.Dataframe(
                    headers=["Original Entity", "Type", "NER Confidence", "Normalized Form", "Match Score", "Alternatives"],
                    label="Linked Entities",
                    wrap=True
                )
            
            with gr.Column(scale=1):
                json_output = gr.JSON(label="Detailed Results")
        
        # Examples section
        # gr.HTML('<div class="stage-header">📋 Example Inputs</div>')
        # gr.Examples(
        #     examples=[[ex] for ex in EXAMPLE_TEXTS],
        #     inputs=input_text,
        #     label=""
        # )
        
        
        # Event handlers
        submit_btn.click(
            fn=predict_medical_entities,
            inputs=input_text,
            outputs=[highlighted_output, entity_table, json_output]
        )
        
        clear_btn.click(
            fn=lambda: ("", [], [], {}),
            inputs=None,
            outputs=[input_text, highlighted_output, entity_table, json_output]
        )
        
        # Also trigger on Enter key
        input_text.submit(
            fn=predict_medical_entities,
            inputs=input_text,
            outputs=[highlighted_output, entity_table, json_output]
        )
    
    return demo


# MAIN ENTRY POINT

if __name__ == "__main__":
    print("\n" + "="*60)
    print(" Starting Medical NER & Contrastive Retrieval System")
    print("="*60)
    
    # Create and launch interface
    demo = create_interface()
    
    print("\n📍 Launching Gradio interface...")
    print("   Open your browser to: http://localhost:7860")
    print("   Press Ctrl+C to stop the server\n")
    
    demo.launch(
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860,
        share=False,  # Set to True to create a public link
        show_error=True,
        css=custom_css
    )
