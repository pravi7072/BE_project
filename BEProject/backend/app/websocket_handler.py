# backend/app/websocket_handler.py
"""
WebSocket handler for real-time audio streaming
Handles bidirectional audio streaming with the frontend
"""

import asyncio
import json
import numpy as np
import torch
from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, Optional
import base64
import time

class WebSocketHandler:
    """Handle real-time audio streaming via WebSocket"""
    
    def __init__(self, config, model_manager):
        """
        Initialize WebSocket handler
        
        Args:
            config: Configuration object
            model_manager: Model manager for inference
        """
        self.config = config
        self.model_manager = model_manager
        
        # Active connections dictionary
        self.active_connections: Dict[str, Dict] = {}
        
        # Import audio processing modules
        from .preprocessing.audio_processor import AudioProcessor
        from .preprocessing.feature_extractor import FeatureExtractor
        from .preprocessing.stream_buffer import StreamBuffer
        
        # Initialize processors
        self.audio_processor = AudioProcessor(config)
        self.feature_extractor = FeatureExtractor(config)
        
        # Store StreamBuffer class for per-client instantiation
        self.StreamBuffer = StreamBuffer
    
    async def connect(self, websocket: WebSocket, client_id: str):
        """
        Accept new WebSocket connection and initialize session
        
        Args:
            websocket: WebSocket connection
            client_id: Unique client identifier
        """
        await websocket.accept()
        
        # Initialize session for this client
        self.active_connections[client_id] = {
            'websocket': websocket,
            'buffer': self.StreamBuffer(
                max_size=self.config.audio.max_buffer_size,
                overlap=self.config.audio.overlap
            ),
            'context': None,
            'speaker_enrolled': False,
            'total_processed': 0,
            'chunks_processed': 0,
            'session_start': time.time(),
            'last_activity': time.time()
        }
        
        print(f"✓ Client {client_id} connected. Active: {len(self.active_connections)}")
        
        # Send initial configuration to client
        await self.send_config(websocket)
    
    def disconnect(self, client_id: str):
        """
        Clean up and remove connection
        
        Args:
            client_id: Client to disconnect
        """
        if client_id in self.active_connections:
            session = self.active_connections[client_id]
            duration = time.time() - session['session_start']
            
            print(f"✗ Client {client_id} disconnected.")
            print(f"  Session duration: {duration:.2f}s")
            print(f"  Samples processed: {session['total_processed']}")
            print(f"  Chunks processed: {session['chunks_processed']}")
            
            # Clean up
            session['buffer'].clear()
            del self.active_connections[client_id]
    
    async def send_config(self, websocket: WebSocket):
        """
        Send audio configuration to client
        
        Args:
            websocket: WebSocket connection
        """
        config_msg = {
            'type': 'config',
            'sample_rate': self.config.audio.sample_rate,
            'chunk_size': self.config.audio.chunk_size,
            'channels': 1,
            'format': 'int16'
        }
        await websocket.send_json(config_msg)
        print(f"  → Sent config: {config_msg}")
    
    async def handle_audio_chunk(self, client_id: str, audio_data: bytes):
        """
        Process incoming audio chunk from client
        
        Args:
            client_id: Client identifier
            audio_data: Raw audio bytes
        """
        if client_id not in self.active_connections:
            print(f"⚠ Client {client_id} not in active connections")
            return
        
        session = self.active_connections[client_id]
        
        try:
            # Update last activity
            session['last_activity'] = time.time()
            
            # Decode audio data (16-bit PCM)
            audio_array = self._decode_audio(audio_data)
            
            # Add to buffer
            session['buffer'].add(audio_array)
            
            # Process if enough data available
            if session['buffer'].is_ready(self.config.audio.chunk_size):
                audio_chunk = session['buffer'].get_chunk(self.config.audio.chunk_size)
                
                # Preprocess audio
                audio_chunk = self.audio_processor.preprocess_pipeline(
                    audio_chunk, 
                    remove_silence=False,  # Keep silence in streaming
                    normalize=True,
                    denoise=False  # Disable for speed
                )
                
                # Extract mel spectrogram
                mel_chunk = self.feature_extractor.extract_mel(
                    torch.FloatTensor(audio_chunk)
                )
                if mel_chunk.dim() == 2:
                    mel_chunk = mel_chunk.unsqueeze(0)
                # Convert to clear speech
                start_time = time.time()
                audio_clear, session['context'] = self.model_manager.convert_streaming(
                    mel_chunk,
                    context=session['context']
                )
                inference_time = time.time() - start_time
                
                # Convert to numpy
                audio_clear_np = audio_clear.squeeze().detach().cpu().numpy()
                audio_clear_np = np.nan_to_num(audio_clear_np)
                audio_clear_np = np.clip(audio_clear_np, -1.0, 1.0)
                # Apply de-emphasis filter
                audio_clear_np = self.audio_processor.apply_deemphasis(audio_clear_np)
                
                # Encode audio for sending
                encoded_audio = self._encode_audio(audio_clear_np)
                
                # Send processed audio back to client
                await self.send_audio(
                    session['websocket'],
                    encoded_audio,
                    inference_time
                )
                
                # Update statistics
                session['total_processed'] += len(audio_chunk)
                session['chunks_processed'] += 1
        
        except Exception as e:
            print(f"✗ Error processing audio for {client_id}: {e}")
            import traceback
            traceback.print_exc()
            await self.send_error(session['websocket'], str(e))
    
    async def handle_speaker_enrollment(self, client_id: str, audio_data: bytes):
        """
        Enroll speaker from provided audio sample
        
        Args:
            client_id: Client identifier
            audio_data: Audio data for enrollment
        """
        if client_id not in self.active_connections:
            return
        
        session = self.active_connections[client_id]
        
        try:
            # Decode audio
            audio_array = self._decode_audio(audio_data)
            
            # Preprocess
            audio_array = self.audio_processor.preprocess_pipeline(audio_array)
            
            # Extract mel
            mel = self.feature_extractor.extract_mel(torch.FloatTensor(audio_array))
            
            # Move to device
            mel = mel.to(self.config.device)
            if self.config.use_half_precision:
                mel = mel.half()
            
            # Extract speaker embedding
            with torch.no_grad():
                speaker_emb = self.model_manager.speaker_encoder(mel)
            
            # Store in session context
            if session['context'] is None:
                session['context'] = {}
            session['context']['speaker_emb'] = speaker_emb
            session['speaker_enrolled'] = True
            
            print(f"✓ Speaker enrolled for client {client_id}")
            
            # Send confirmation
            await session['websocket'].send_json({
                'type': 'enrollment_complete',
                'message': 'Speaker enrolled successfully',
                'timestamp': time.time()
            })
            
        except Exception as e:
            print(f"✗ Error enrolling speaker for {client_id}: {e}")
            await self.send_error(session['websocket'], str(e))
    
    async def send_audio(self, websocket: WebSocket, audio_data: bytes, 
                        inference_time: float):
        """
        Send processed audio to client
        
        Args:
            websocket: WebSocket connection
            audio_data: Processed audio bytes
            inference_time: Time taken for inference
        """
        message = {
            'type': 'audio',
            'data': base64.b64encode(audio_data).decode('utf-8'),
            'inference_time_ms': inference_time * 1000,
            'timestamp': time.time()
        }
        await websocket.send_json(message)
    
    async def send_error(self, websocket: WebSocket, error_msg: str):
        """
        Send error message to client
        
        Args:
            websocket: WebSocket connection
            error_msg: Error message
        """
        message = {
            'type': 'error',
            'message': error_msg,
            'timestamp': time.time()
        }
        await websocket.send_json(message)
    
    def _decode_audio(self, audio_data: bytes) -> np.ndarray:
        """
        Decode audio bytes to numpy array
        
        Args:
            audio_data: Raw audio bytes (16-bit PCM)
            
        Returns:
            Normalized audio array [-1, 1]
        """
        # Convert bytes to int16 array
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
        
        # Normalize to [-1, 1]
        audio_array = audio_array / 32768.0
        
        return audio_array
    
    def _encode_audio(self, audio_array: np.ndarray) -> bytes:
        """
        Encode numpy array to audio bytes
        
        Args:
            audio_array: Normalized audio array [-1, 1]
            
        Returns:
            Audio bytes (16-bit PCM)
        """
        # Clip to valid range
        audio_array = np.clip(audio_array, -1.0, 1.0)
        
        # Convert to 16-bit PCM
        audio_int16 = (audio_array * 32767).astype(np.int16)
        
        return audio_int16.tobytes()
    
    async def handle_message(self, client_id: str, message: dict):
        """
        Route incoming WebSocket messages
        
        Args:
            client_id: Client identifier
            message: Parsed JSON message
        """
        msg_type = message.get('type')
        
        if msg_type == 'audio':
            # Handle audio chunk
            audio_data = base64.b64decode(message['data'])
            await self.handle_audio_chunk(client_id, audio_data)
        
        elif msg_type == 'enroll':
            # Handle speaker enrollment
            audio_data = base64.b64decode(message['data'])
            await self.handle_speaker_enrollment(client_id, audio_data)
        
        elif msg_type == 'reset':
            # Reset session
            if client_id in self.active_connections:
                session = self.active_connections[client_id]
                session['buffer'].clear()
                session['context'] = None
                session['speaker_enrolled'] = False
                session['total_processed'] = 0
                session['chunks_processed'] = 0
                
                await session['websocket'].send_json({
                    'type': 'reset_complete',
                    'timestamp': time.time()
                })
                print(f"✓ Session reset for client {client_id}")
        
        elif msg_type == 'ping':
            # Health check / keep-alive
            if client_id in self.active_connections:
                await self.active_connections[client_id]['websocket'].send_json({
                    'type': 'pong',
                    'timestamp': time.time()
                })
        
        else:
            print(f"⚠ Unknown message type: {msg_type}")
    
    def get_stats(self) -> dict:
        """
        Get statistics for all active connections
        
        Returns:
            Dictionary with connection statistics
        """
        stats = {
            'total_connections': len(self.active_connections),
            'connections': []
        }
        
        for client_id, session in self.active_connections.items():
            uptime = time.time() - session['session_start']
            idle_time = time.time() - session['last_activity']
            
            stats['connections'].append({
                'client_id': client_id,
                'uptime_seconds': uptime,
                'idle_seconds': idle_time,
                'total_processed': session['total_processed'],
                'chunks_processed': session['chunks_processed'],
                'speaker_enrolled': session['speaker_enrolled'],
                'buffer_size': len(session['buffer']),
                'buffer_fill': session['buffer'].fill_percentage
            })
        
        return stats
