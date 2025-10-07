import torch
import numpy as np
import time
import platform

def test_auracore_hardware():
    """Test RTX 4050 setup for AuraCore project"""
    print("=== AURACORE HARDWARE VERIFICATION ===")
    print(f"System: {platform.system()} {platform.release()}")
    print(f"Python: {platform.python_version()}")
    
    # Check CUDA availability
    print(f"\n🔍 CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"🎯 GPU: {device_name}")
        
        # Verify RTX 4050
        if "4050" in device_name:
            print("✅ RTX 4050 detected - Perfect for AuraCore!")
        else:
            print(f"⚠️  Detected: {device_name} (Project will adapt accordingly)")
            
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🔋 VRAM: {memory_gb:.1f} GB")
        print(f"🔧 CUDA Version: {torch.version.cuda}")
        
        # Memory allocation test
        try:
            # Test with 1GB allocation (safe for 6GB card)
            test_tensor = torch.randn(1024, 1024, 32).cuda()
            print(f"✅ GPU Memory Test Passed ({test_tensor.element_size() * test_tensor.numel() / 1024**3:.2f} GB allocated)")
            del test_tensor
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"❌ GPU Memory Test Failed: {e}")
    
    # Performance benchmark for audio processing
    print(f"\n=== AURACORE PERFORMANCE BENCHMARK ===")
    if torch.cuda.is_available():
        # Audio buffer simulation (typical 1024 samples)
        buffer_size = 1024
        channels = 2
        batch_size = 32  # Process multiple buffers at once
        
        # Create test audio buffers
        audio_buffers = torch.randn(batch_size, channels, buffer_size).cuda()
        
        # Test convolution (typical for feature extraction)
        conv1d = torch.nn.Conv1d(channels, 64, kernel_size=15, padding=7).cuda()
        
        # Warm-up
        for _ in range(10):
            _ = conv1d(audio_buffers)
        torch.cuda.synchronize()
        
        # Actual benchmark
        start_time = time.time()
        for _ in range(100):
            features = conv1d(audio_buffers)
        torch.cuda.synchronize()
        gpu_time = time.time() - start_time
        
        samples_per_second = (buffer_size * batch_size * 100) / gpu_time
        print(f"🚀 GPU Processing Speed: {samples_per_second/44100:.1f}x real-time")
        print(f"⏱️  Latency per buffer: {(gpu_time/100)*1000:.2f}ms")
        
        if samples_per_second > 44100 * 5:  # 5x real-time minimum
            print("✅ GPU Performance: Excellent for real-time processing!")
        else:
            print("⚠️  GPU Performance: May need optimization for real-time")
    
    # Test audio processing libraries
    print(f"\n=== AUDIO LIBRARIES TEST ===")
    try:
        import librosa
        import soundfile as sf
        
        # Create test signal (1 second of audio)
        sr = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))
        test_signal = np.sin(2 * np.pi * 440 * t) * 0.5  # 440 Hz sine wave
        
        print(f"📊 Test signal: {len(test_signal)} samples @ {sr}Hz")
        
        # Feature extraction benchmark
        start_time = time.time()
        mfcc = librosa.feature.mfcc(y=test_signal, sr=sr, n_mfcc=13)
        spectral_centroid = librosa.feature.spectral_centroid(y=test_signal, sr=sr)
        chroma = librosa.feature.chroma_stft(y=test_signal, sr=sr)
        feature_time = time.time() - start_time
        
        print(f"⚡ Feature extraction: {feature_time*1000:.1f}ms")
        print(f"📈 MFCC shape: {mfcc.shape}")
        print(f"🎵 Features extracted successfully")
        
        if feature_time < 0.1:  # Under 100ms for 1 second of audio
            print("✅ Audio processing: Fast enough for real-time!")
        else:
            print("⚠️  Audio processing: May need optimization")
            
    except ImportError as e:
        print(f"❌ Audio library missing: {e}")
        print("Run: pip install librosa soundfile")
    except Exception as e:
        print(f"❌ Audio processing error: {e}")
    
    # Final assessment
    print(f"\n=== AURACORE FEASIBILITY ASSESSMENT ===")
    if torch.cuda.is_available():
        print("✅ Hardware: Ready for AI audio processing")
        print("🎯 Recommendation: Start with lightweight models, optimize later")
        print("🚀 Expected latency: 10-50ms (acceptable for most applications)")
    else:
        print("⚠️  Hardware: CPU-only mode (still functional, but slower)")
        print("🎯 Recommendation: Focus on algorithmic innovation over speed")
    
    print(f"\n🎵 AURACORE PROJECT: Ready to begin! 🎵")

if __name__ == "__main__":
    test_auracore_hardware()