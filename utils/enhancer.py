import os
import cv2
import numpy as np
from PIL import Image
import torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Import dengan error handling
try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    print(" Real-ESRGAN dependencies imported successfully")
except ImportError as e:
    print(f" Import Error: {e}")
    print("Install dependencies:")
    print("pip install realesrgan")
    print("pip install basicsr")
    raise

class FaceEnhancer:
    def __init__(self, model_name='RealESRGAN_x4plus', model_path=None, scale=4, tile=400, half=True):
        """
        Inisialisasi Real-ESRGAN untuk enhancement wajah dengan error handling yang lebih baik
        """
        self.model_name = model_name
        self.scale = scale
        self.tile = tile
        self.half = half
        self.upsampler = None
        
        self.device = self._check_device()
        print(f"Using device: {self.device}")

        try:
            self._setup_model(model_name, model_path)
        except Exception as e:
            print(f" Failed to setup model: {e}")
            print("Detailed error info:")
            import traceback
            traceback.print_exc()
            raise
        
    def _check_device(self):
        """Check device availability"""
        if torch.cuda.is_available():
            try:
                torch.cuda.init()
                device_name = torch.cuda.get_device_name(0)
                memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f" CUDA available: {device_name}")
                print(f" CUDA memory: {memory:.1f} GB")
                return 'cuda'
            except Exception as e:
                print(f" CUDA available but failed to initialize: {e}")
                return 'cpu'
        else:
            print(" CUDA not available, using CPU")
            return 'cpu'
        
    def _find_model_path(self, model_name):
        """Mencari path model yang tersedia"""
        # Possible model locations
        possible_paths = [
            f"models/{model_name}.pth",
            f"./models/{model_name}.pth", 
            f"../models/{model_name}.pth",
            f"weights/{model_name}.pth",
            f"./weights/{model_name}.pth",
            f"{model_name}.pth",
            f"./{model_name}.pth"
        ]
        
        print(f"Mencari model {model_name}...")
        for path in possible_paths:
            if os.path.exists(path):
                abs_path = os.path.abspath(path)
                print(f" Model ditemukan di: {abs_path}")
                return abs_path
                
        print(f" Model file tidak ditemukan di lokasi lokal")
        print("Model akan di-download otomatis oleh Real-ESRGAN")
        return None
        
    def _setup_model(self, model_name, model_path):
        """Setup model dengan error handling yang lebih komprehensif"""
        
        if self.device == 'cpu':
            self.half = False
            self.tile = min(self.tile, 200)
            print(" CPU mode: disabled half precision, reduced tile size")
        
        if model_path is None:
            model_path = self._find_model_path(model_name)
        
        try:
            print(f"Configuring model architecture for {model_name}...")
            
            if 'anime_6B' in model_name:
                model = RRDBNet(
                    num_in_ch=3, 
                    num_out_ch=3, 
                    num_feat=64, 
                    num_block=6, 
                    num_grow_ch=32, 
                    scale=self.scale
                )
                print(" Using anime model architecture (6 blocks)")
            elif 'x2' in model_name:
                model = RRDBNet(
                    num_in_ch=3, 
                    num_out_ch=3, 
                    num_feat=64, 
                    num_block=23, 
                    num_grow_ch=32, 
                    scale=2
                )
                self.scale = 2
                print(" Using x2 model architecture")
            else:
                model = RRDBNet(
                    num_in_ch=3, 
                    num_out_ch=3, 
                    num_feat=64, 
                    num_block=23, 
                    num_grow_ch=32, 
                    scale=self.scale
                )
                print(" Using standard model architecture")
            
            print(f"Initializing RealESRGANer...")
            print(f"  - Model path: {model_path if model_path else 'Auto-download'}")
            print(f"  - Scale: {self.scale}x")
            print(f"  - Tile: {self.tile}")
            print(f"  - Half: {self.half}")
            
            # Inisialisasi dengan parameter yang lebih aman
            init_params = {
                'scale': self.scale,
                'model_path': model_path,
                'dni_weight': None,
                'model': model,
                'tile': self.tile,
                'tile_pad': 10,
                'pre_pad': 0,
                'half': self.half
            }
            
            # Tambahkan gpu_id hanya jika menggunakan CUDA
            if self.device == 'cuda':
                init_params['gpu_id'] = 0
            else:
                init_params['gpu_id'] = None
            
            # Inisialisasi upsampler
            self.upsampler = RealESRGANer(**init_params)
            
            print(f" Real-ESRGAN model '{model_name}' loaded successfully!")
            print(f" Configuration summary:")
            print(f"  - Scale: {self.scale}x")
            print(f"  - Tile size: {self.tile}")
            print(f"  - Half precision: {self.half}")
            print(f"  - Device: {self.device}")
            
        except Exception as e:
            print(f" Error during model initialization: {type(e).__name__}: {e}")
            
            # Specific error handling
            if "startswith" in str(e):
                print("\n Fix: Model path issue detected")
                print("Trying to initialize without model_path...")
                try:
                    # Coba tanpa model_path, biarkan auto-download
                    init_params['model_path'] = None
                    self.upsampler = RealESRGANer(**init_params)
                    print(" Model initialized successfully with auto-download")
                except Exception as e2:
                    print(f" Still failed: {e2}")
                    raise
            else:
                print("\n Troubleshooting suggestions:")
                print("1. Check if model file exists and is valid")
                print("2. Try reducing tile size (e.g., tile=200)")
                print("3. Disable half precision (half=False)")
                print("4. Check CUDA installation")
                print("5. Try CPU mode")
                raise
    
    def enhance_image(self, image):
        """Enhancement gambar dengan error handling yang comprehensive"""
        if self.upsampler is None:
            print(" Model not initialized")
            return image, False
            
        try:
            # Input validation dan konversi
            if isinstance(image, Image.Image):
                image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            if image is None or image.size == 0:
                print(" Invalid input image")
                return image, False
            
            # Cek dimensi minimum
            h, w = image.shape[:2]
            if h < 4 or w < 4:
                print(f" Image too small ({w}x{h}), skipping enhancement")
                return image, False
            
            print(f"  Processing image: {w}x{h} -> {w*self.scale}x{h*self.scale}")
            
            # Enhancement dengan error handling
            try:
                enhanced_img, _ = self.upsampler.enhance(image, outscale=self.scale)
                return enhanced_img, True
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f" CUDA out of memory, trying with smaller tile size")
                    # Coba dengan tile size yang lebih kecil
                    original_tile = self.tile
                    self.upsampler.tile = max(100, self.tile // 2)
                    try:
                        enhanced_img, _ = self.upsampler.enhance(image, outscale=self.scale)
                        print(f" Success with reduced tile size: {self.upsampler.tile}")
                        return enhanced_img, True
                    except:
                        # Restore original tile size
                        self.upsampler.tile = original_tile
                        raise
                else:
                    raise
            
        except Exception as e:
            print(f" Enhancement failed: {type(e).__name__}: {e}")
            return image, False
    
    def enhance_face_crop(self, crop_path, output_path, quality=95):
        """Enhancement single face crop dengan logging yang lebih baik"""
        enhancement_info = {
            'original_path': crop_path,
            'output_path': output_path,
            'original_size': None,
            'enhanced_size': None,
            'scale_factor': self.scale,
            'success': False
        }
        
        try:
            # Baca gambar dengan validasi
            if not os.path.exists(crop_path):
                print(f" File not found: {crop_path}")
                return False, enhancement_info
                
            img = cv2.imread(crop_path, cv2.IMREAD_COLOR)
            if img is None:
                print(f" Failed to read image: {crop_path}")
                return False, enhancement_info
            
            enhancement_info['original_size'] = (img.shape[1], img.shape[0])
            
            # Enhancement
            enhanced_img, success = self.enhance_image(img)
            
            if not success:
                return False, enhancement_info
            
            enhancement_info['enhanced_size'] = (enhanced_img.shape[1], enhanced_img.shape[0])
            
            # Buat direktori output jika belum ada
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Simpan hasil
            _, ext = os.path.splitext(output_path)
            if ext.lower() in ['.jpg', '.jpeg']:
                success_save = cv2.imwrite(output_path, enhanced_img, 
                                         [cv2.IMWRITE_JPEG_QUALITY, quality])
            else:
                success_save = cv2.imwrite(output_path, enhanced_img)
            
            if not success_save:
                print(f" Failed to save enhanced image: {output_path}")
                return False, enhancement_info
            
            enhancement_info['success'] = True
            return True, enhancement_info
            
        except Exception as e:
            print(f" Error enhancing {os.path.basename(crop_path)}: {e}")
            return False, enhancement_info
    
    def get_model_info(self):
        """Info model yang lebih lengkap"""
        return {
            'model_name': self.model_name,
            'scale': self.scale,
            'tile': self.tile,
            'half_precision': self.half,
            'device': self.device,
            'is_loaded': self.upsampler is not None,
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available()
        }

# Batch enhancement function yang diperbaiki
def enhance_face_crops_batch(crops_dir, enhancer, prefix="enhanced", progress_callback=None):
    """Enhanced batch processing dengan error recovery yang lebih baik"""
    
    results = {
        'enhanced_files': [],
        'failed_files': [],
        'enhancement_info': [],
        'statistics': {
            'total_files': 0,
            'successful': 0,
            'failed': 0,
            'total_time': 0
        }
    }
    
    if not os.path.exists(crops_dir):
        print(f" Crops directory not found: {crops_dir}")
        return results
    
    # Buat direktori enhanced
    enhanced_dir = os.path.join(os.path.dirname(crops_dir), f"{prefix}_enhanced")
    os.makedirs(enhanced_dir, exist_ok=True)
    print(f" Enhanced files will be saved to: {enhanced_dir}")
    
    # Ambil file gambar
    crop_files = [f for f in os.listdir(crops_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    results['statistics']['total_files'] = len(crop_files)
    
    if not crop_files:
        print(" No image files found in crops directory")
        return results
    
    print(f"\n Starting batch enhancement of {len(crop_files)} files...")
    
    import time
    start_time = time.time()
    
    for i, crop_file in enumerate(crop_files, 1):
        crop_path = os.path.join(crops_dir, crop_file)
        
        # Generate output filename
        name, ext = os.path.splitext(crop_file)
        enhanced_filename = f"{prefix}_{name}{ext}"
        enhanced_path = os.path.join(enhanced_dir, enhanced_filename)
        
        print(f"\n[{i}/{len(crop_files)}] Processing: {crop_file}")
        
        # Progress callback
        if progress_callback:
            try:
                progress_callback(i, len(crop_files), crop_file)
            except: 
                pass  # Ignore callback errors
        
        # Enhancement dengan retry mechanism
        max_retries = 2
        success = False
        info = None
        
        for attempt in range(max_retries):
            if attempt > 0:
                print(f"   Retry attempt {attempt}/{max_retries-1}")
            
            try:
                success, info = enhancer.enhance_face_crop(crop_path, enhanced_path)
                if success:
                    break
            except Exception as e:
                print(f"   Attempt {attempt+1} failed: {e}")
                if attempt == max_retries - 1:
                    success = False
        
        # Record results
        if success and info:
            results['enhanced_files'].append(enhanced_path)
            results['enhancement_info'].append(info)
            results['statistics']['successful'] += 1
            
            orig_size = info['original_size']
            enh_size = info['enhanced_size']
            print(f"   Success: {orig_size[0]}x{orig_size[1]} → {enh_size[0]}x{enh_size[1]}")
        else:
            results['failed_files'].append(crop_path)
            results['statistics']['failed'] += 1
            print(f"   Failed: {crop_file}")
    
    results['statistics']['total_time'] = time.time() - start_time
    
    # Print final summary
    stats = results['statistics']
    print(f"\n{'='*50}")
    print(f" BATCH ENHANCEMENT COMPLETE")
    print(f"{ '='*50}")
    print(f" Total files: {stats['total_files']}")
    print(f" Successful: {stats['successful']}")
    print(f" Failed: {stats['failed']}")
    print(f" Success rate: {(stats['successful']/stats['total_files']*100):.1f}%")
    print(f" Total time: {stats['total_time']:.2f} seconds")
    print(f" Enhanced files saved to: {enhanced_dir}")
    
    return results

def create_enhancement_summary(results, output_path):
    """Membuat summary file yang lebih informatif"""
    stats = results['statistics']
    
    summary = f"""
=== LAPORAN ENHANCEMENT WAJAH ===
Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

--- RINGKASAN STATISTIK ---
Total File Diproses: {stats['total_files']}
 Berhasil: {stats['successful']}
 Gagal: {stats['failed']}
 Tingkat Keberhasilan: {(stats['successful']/max(stats['total_files'], 1)*100):.1f}%
 Waktu Total: {stats['total_time']:.2f} detik
 Waktu Rata-rata per File: {(stats['total_time']/max(stats['total_files'], 1)):.2f} detik

--- DETAIL FILE BERHASIL ---
"""
    
    for i, info in enumerate(results['enhancement_info'], 1):
        orig_size = info['original_size']
        enh_size = info['enhanced_size']
        filename = os.path.basename(info['original_path'])
        
        summary += f"\nFile #{i}: {filename}\n"
        summary += f"   Ukuran Asli: {orig_size[0]}x{orig_size[1]} px\n"
        summary += f"   Ukuran Enhanced: {enh_size[0]}x{enh_size[1]} px\n"
        summary += f"   Scale Factor: {info['scale_factor']}x\n"
        summary += f"   Output: {os.path.basename(info['output_path'])}\n"
    
    if results['failed_files']:
        summary += f"\n--- FILE GAGAL ({len(results['failed_files'])}) ---\n"
        for i, failed_file in enumerate(results['failed_files'], 1):
            filename = os.path.basename(failed_file)
            summary += f"{i}. {filename}\n"
    
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        print(f" Enhancement summary saved to: {os.path.basename(output_path)}")
    except Exception as e:
        print(f" Error saving enhancement summary: {e}")

# Utility functions yang diperbaiki
def get_available_models():
    """Return available Real-ESRGAN models dengan info lebih lengkap"""
    return {
        'RealESRGAN_x4plus': {
            'description': 'Model utama untuk gambar natural (4x)',
            'scale': 4,
            'best_for': 'Foto natural, potret, wajah',
            'file_size': '~65MB',
            'recommended_tile': 400
        },
        'RealESRGAN_x2plus': {
            'description': 'Model untuk enhancement 2x (lebih cepat)',
            'scale': 2,
            'best_for': 'Enhancement ringan, GPU terbatas',
            'file_size': '~65MB', 
            'recommended_tile': 600
        },
        'RealESRGAN_x4plus_anime_6B': {
            'description': 'Model khusus untuk anime/kartun (4x)',
            'scale': 4,
            'best_for': 'Gambar anime, kartun, ilustrasi',
            'file_size': '~18MB',
            'recommended_tile': 400
        }
    }
