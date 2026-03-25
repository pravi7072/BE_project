
# backend/app/main.py
"""
FastAPI main application
Serves REST API and WebSocket endpoints
"""
import json  # Add this line after other imports
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import uvicorn
import uuid
import os
import torch
import tempfile
from pathlib import Path
from typing import Optional

# Import application components
from .utils.config import Config
from .models.model_manager import ModelManager
from .websocket_handler import WebSocketHandler

# Initialize configuration
config = Config()

# Initialize FastAPI app
app = FastAPI(
    title="Dysarthric Speech Conversion API",
    description="Real-time dysarthric to clear speech conversion with WebSocket support",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.server.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model_manager: Optional[ModelManager] = None
ws_handler: Optional[WebSocketHandler] = None

@app.on_event("startup")
async def startup_event():
    """Initialize models and services on startup"""
    global model_manager, ws_handler
    
    print("=" * 60)
    print("Starting Dysarthric Speech Conversion Server...")
    print("=" * 60)
    
    try:
        # Find checkpoint
        checkpoint_dir = config.paths.checkpoint_dir
        checkpoint_path = None
        
        if Path(checkpoint_dir).exists():
            best_model = Path(checkpoint_dir) / 'best_model.pt'
            if best_model.exists():
                checkpoint_path = str(best_model)
                print(f"✓ Found best model: {checkpoint_path}")
            else:
                # Look for latest checkpoint
                checkpoints = sorted(Path(checkpoint_dir).glob('checkpoint_epoch_*.pt'))
                if checkpoints:
                    checkpoint_path = str(checkpoints[-1])
                    print(f"✓ Found checkpoint: {checkpoint_path}")
        
        if checkpoint_path is None:
            print("⚠ No checkpoint found - using randomly initialized models")
            print("  Train a model first: python scripts/train.py")
        
        # Initialize model manager
        print("\nInitializing models...")
        model_manager = ModelManager(config, checkpoint_path)
        
        # Print model info
        model_info = model_manager.get_model_info()
        print("\nModel Information:")
        for key, value in model_info.items():
            print(f"  {key}: {value}")
        
        # Initialize WebSocket handler
        print("\nInitializing WebSocket handler...")
        ws_handler = WebSocketHandler(config, model_manager)
        
        print("\n" + "=" * 60)
        print("✓ Server ready!")
        print(f"  Device: {config.device}")
        print(f"  Port: {config.server.port}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Error during startup: {e}")
        import traceback
        traceback.print_exc()
        raise

@app.get("/")
async def root():
    """Health check and basic info"""
    return {
        "status": "running",
        "service": "Dysarthric Speech Conversion",
        "version": "1.0.0",
        "device": str(config.device),
        "active_connections": len(ws_handler.active_connections) if ws_handler else 0
    }

@app.get("/health")
async def health():
    """Detailed health check"""
    if model_manager is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    return {
        "status": "healthy",
        "models_loaded": True,
        "model_info": model_manager.get_model_info(),
        "websocket_connections": len(ws_handler.active_connections) if ws_handler else 0,
        "config": {
            "sample_rate": config.audio.sample_rate,
            "chunk_size": config.audio.chunk_size,
            "device": str(config.device)
        }
    }

@app.post("/convert/file")
async def convert_file(file: UploadFile = File(...)):
    """
    Convert uploaded audio file
    
    Args:
        file: Audio file (WAV format recommended)
        
    Returns:
        Converted audio file
    """
    if model_manager is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded file
            input_path = Path(temp_dir) / f"input_{uuid.uuid4().hex}.wav"
            with open(input_path, "wb") as f:
                content = await file.read()
                f.write(content)
            
            # Load and process
            from .preprocessing.audio_processor import AudioProcessor
            from .preprocessing.feature_extractor import FeatureExtractor
            
            audio_processor = AudioProcessor(config)
            feature_extractor = FeatureExtractor(config)
            
            # Load audio
            audio = audio_processor.preprocess_pipeline(
                audio_processor.load_audio(str(input_path))
            )
            
            mel = feature_extractor.extract_mel(torch.FloatTensor(audio))

            if mel.dim() == 2:
                mel = mel.unsqueeze(0)

            mel = mel.to(config.device)

            audio_clear = model_manager.convert(mel)
            
            # Post-process
            audio_clear_np = audio_clear.squeeze().cpu().numpy()
            audio_clear_np = audio_processor.apply_deemphasis(audio_clear_np)
            
            # Save output
            output_path = Path(temp_dir) / f"output_{uuid.uuid4().hex}.wav"
            audio_processor.save_audio(audio_clear_np, str(output_path))
            
            # Return file
            return FileResponse(
                output_path,
                media_type="audio/wav",
                filename="converted_clear.wav"
            )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """
    WebSocket endpoint for real-time streaming
    
    Args:
        websocket: WebSocket connection
        client_id: Unique client identifier
    """
    if ws_handler is None:
        await websocket.close(code=1011, reason="Server not ready")
        return
    
    await ws_handler.connect(websocket, client_id)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data) if isinstance(data, str) else data
            
            # Handle message
            await ws_handler.handle_message(client_id, message)
    
    except WebSocketDisconnect:
        ws_handler.disconnect(client_id)
    except Exception as e:
        print(f"WebSocket error for {client_id}: {e}")
        ws_handler.disconnect(client_id)

@app.get("/stats")
async def get_stats():
    """Get server statistics"""
    if ws_handler is None:
        return {"error": "Handler not initialized"}
    
    return ws_handler.get_stats()

def main():
    """Run the server"""
    uvicorn.run(
        "backend.app.main:app",
        host=config.server.host,
        port=config.server.port,
        reload=False,
        log_level="info"
    )

if __name__ == "__main__":
    import json  # Add import for json
    main()
