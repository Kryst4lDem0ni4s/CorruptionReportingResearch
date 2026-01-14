# "faceforensics/
# ├── real/ # Real images
# ├── fake/ # Deepfake images
# └── README.md # This file

# text

# ## Citation

# @inproceedings{roessler2019faceforensicspp,
# title={FaceForensics++: Learning to Detect Manipulated Facial Images},
# author={Rössler, Andreas and Cozzolino, Davide and Verdoliva, Luisa and Riess, Christian and Thies, Justus and Nießner, Matthias},
# booktitle={International Conference on Computer Vision (ICCV)},
# year={2019}
# }

# text

# ## License

# Research use only. See original dataset for full license.
# """
        
#         with open(readme_path, 'w') as f:
#             f.write(readme_content)
        
#         logger.info(f"Created directory structure at: {dataset_path}")
#         logger.info(f"Created README at: {readme_path}")
#         logger.info("")
#         logger.info("NOTE: Dataset download requires manual steps")
#         logger.info("Run this script again after downloading to validate")
        
#         # Check if files already exist
#         real_files = list((dataset_path / 'real').glob('*'))
#         fake_files = list((dataset_path / 'fake').glob('*'))
        
#         if len(real_files) > 0 or len(fake_files) > 0:
#             logger.info(f"Found {len(real_files)} real and {len(fake_files)} fake samples")
#             return True
        
#         return False
    
#     # ========================================
#     # CELEB-DF DOWNLOAD
#     # ========================================
    
#     def _download_celebdf(self, sample_size: Optional[int] = None) -> bool:
#         """
#         Download Celeb-DF v2 dataset
        
#         Note: Celeb-DF requires manual download.
        
#         Args:
#             sample_size: Number of samples to download
            
#         Returns:
#             True if successful
#         """
#         logger.info("Celeb-DF v2 Download Instructions")
#         logger.info("="*60)
#         logger.info("")
#         logger.info("Celeb-DF v2 requires manual download:")
#         logger.info("")
#         logger.info("1. Visit: https://github.com/yuezunli/celeb-deepfakeforensics")
#         logger.info("2. Fill out the request form")
#         logger.info("3. Download the dataset")
#         logger.info("4. Extract to: {}".format(get_dataset_path('celebdf')))
#         logger.info("")
#         logger.info("Recommended subset for testing:")
#         logger.info("  - Celeb-real: 50 videos")
#         logger.info("  - Celeb-synthesis: 50 videos")
#         logger.info("  - Size: ~5GB")
#         logger.info("")
#         logger.info("="*60)
        
#         # Create directory structure
#         dataset_path = get_dataset_path('celebdf')
#         (dataset_path / 'Celeb-real').mkdir(exist_ok=True)
#         (dataset_path / 'Celeb-synthesis').mkdir(exist_ok=True)
#         (dataset_path / 'YouTube-real').mkdir(exist_ok=True)
        
#         # Create README
#         readme_path = dataset_path / 'README.md'
#         readme_content = """# Celeb-DF v2 Dataset

# ## Download Instructions

# 1. Visit: https://github.com/yuezunli/celeb-deepfakeforensics
# 2. Fill out dataset request form
# 3. Download Celeb-DF v2
# 4. Extract videos to this directory

# ## Directory Structure

# celebdf/
# ├── Celeb-real/ # Real celebrity videos
# ├── Celeb-synthesis/ # Deepfake videos
# ├── YouTube-real/ # YouTube videos
# └── README.md # This file

# text

# ## Citation

# @inproceedings{li2020celeb,
# title={Celeb-df: A large-scale challenging dataset for deepfake forensics},
# author={Li, Yuezun and Yang, Xin and Sun, Pu and Qi, Honggang and Lyu, Siwei},
# booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
# pages={3207--3216},
# year={2020}
# }

# text

# ## License

# Research use only. See original dataset for full license.
# """
        
#         with open(readme_path, 'w') as f:
#             f.write(readme_content)
        
#         logger.info(f"Created directory structure at: {dataset_path}")
#         logger.info(f"Created README at: {readme_path}")
#         logger.info("")
#         logger.info("NOTE: Dataset download requires manual steps")
        
#         # Check if files exist
#         real_files = list((dataset_path / 'Celeb-real').glob('*'))
#         fake_files = list((dataset_path / 'Celeb-synthesis').glob('*'))
        
#         if len(real_files) > 0 or len(fake_files) > 0:
#             logger.info(f"Found {len(real_files)} real and {len(fake_files)} fake videos")
#             return True
        
#         return False
    
#     # ========================================
#     # SYNTHETIC ATTACKS
#     # ========================================
    
#     def _generate_synthetic_attacks(self, num_groups: Optional[int] = None) -> bool:
#         """
#         Generate synthetic coordinated attack data
        
#         Args:
#             num_groups: Number of attack groups to generate
            
#         Returns:
#             True if successful
#         """
#         logger.info("Generating synthetic coordinated attacks...")
        
#         num_groups = num_groups or 10
#         dataset_path = get_dataset_path('synthetic_attacks')
        
#         # Create directory
#         dataset_path.mkdir(exist_ok=True)
        
#         # Generate attack scenarios
#         attacks = []
#         for i in range(num_groups):
#             attack = {
#                 'id': f'attack_{i:03d}',
#                 'group_size': 3 + (i % 8),  # 3-10 submissions
#                 'pattern': 'linguistic_similarity' if i % 2 == 0 else 'temporal_clustering',
#                 'similarity_score': 0.75 + (i % 25) / 100,
#                 'time_window_hours': 2,
#                 'created': datetime.now().isoformat()
#             }
#             attacks.append(attack)
        
#         # Save to JSON
#         output_path = dataset_path / 'synthetic_attacks.json'
#         with open(output_path, 'w') as f:
#             json.dump(attacks, f, indent=2)
        
#         logger.info(f"Generated {num_groups} attack scenarios")
#         logger.info(f"Saved to: {output_path}")
        
#         # Create README
#         readme_path = dataset_path / 'README.md'
#         readme_content = f"""# Synthetic Coordinated Attacks

# ## Description

# Generated synthetic data for testing coordination detection algorithms.

# ## Data Format

# ```json
# {{
#   "id": "attack_001",
#   "group_size": 5,
#   "pattern": "linguistic_similarity",
#   "similarity_score": 0.85,
#   "time_window_hours": 2,
#   "created": "2026-01-14T13:48:00"
# }}
# Statistics
# Number of attack groups: {num_groups}

# Group sizes: 3-10 submissions

# Patterns: linguistic_similarity, temporal_clustering

# Usage
# Use evaluation/datasets/generate_synthetic.py for more advanced generation.

# Generated
# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# """

# text
#     with open(readme_path, 'w') as f:
#         f.write(readme_content)
    
#     return True

# # ========================================
# # UTILITIES
# # ========================================

# def _mark_downloaded(self, dataset_name: str):
#     """Mark dataset as downloaded"""
#     dataset_path = get_dataset_path(dataset_name)
#     marker_file = dataset_path / '.downloaded'
    
#     metadata = {
#         'dataset': dataset_name,
#         'downloaded_at': datetime.now().isoformat(),
#         'stats': get_dataset_statistics(dataset_name)
#     }
    
#     with open(marker_file, 'w') as f:
#         json.dump(metadata, f, indent=2)

# def print_summary(self):
#     """Print download summary"""
#     logger.info("\n" + "="*60)
#     logger.info("Download Summary")
#     logger.info("="*60)
#     logger.info(f"Downloaded: {len(self.stats['downloaded'])} dataset(s)")
#     for dataset in self.stats['downloaded']:
#         logger.info(f"  ✓ {dataset}")
    
#     if self.stats['skipped']:
#         logger.info(f"Skipped: {len(self.stats['skipped'])} dataset(s)")
#         for dataset in self.stats['skipped']:
#             logger.info(f"  - {dataset}")
    
#     if self.stats['failed']:
#         logger.info(f"Failed: {len(self.stats['failed'])} dataset(s)")
#         for dataset in self.stats['failed']:
#             logger.info(f"  ✗ {dataset}")
    
#     logger.info("="*60)
# ============================================
# COMMAND LINE INTERFACE
# ============================================
# def parse_args():
# """Parse command line arguments"""
# parser = argparse.ArgumentParser(
# description='Download evaluation datasets',
# formatter_class=argparse.RawDescriptionHelpFormatter,
# epilog="""
# Examples:

# Download all datasets
# python download_datasets.py

# Download specific dataset
# python download_datasets.py --dataset faceforensics

# Download with sample limit
# python download_datasets.py --dataset faceforensics --samples 100

# List available datasets
# python download_datasets.py --list

# Check download status
# python download_datasets.py --status

# Note: FaceForensics++ and Celeb-DF require manual download.
# This script provides instructions and validates downloads.
# """
# )

# text
# parser.add_argument(
#     '--dataset',
#     type=str,
#     choices=list_datasets(),
#     help='Specific dataset to download'
# )

# parser.add_argument(
#     '--samples',
#     type=int,
#     help='Number of samples to download'
# )

# parser.add_argument(
#     '--force',
#     action='store_true',
#     help='Force re-download'
# )

# parser.add_argument(
#     '--list',
#     action='store_true',
#     help='List available datasets'
# )

# parser.add_argument(
#     '--status',
#     action='store_true',
#     help='Show download status'
# )

# parser.add_argument(
#     '--verbose',
#     action='store_true',
#     help='Enable verbose logging'
# )

# parser.add_argument(
#     '--dry-run',
#     action='store_true',
#     help='Dry run (check availability only)'
# )

# return parser.parse_args()
# ============================================
# MAIN FUNCTION
# ============================================
# def main():
# """Main download function"""
# args = parse_args()

# text
# # Setup logging
# if args.verbose:
#     logger.setLevel(logging.DEBUG)

# # List datasets
# if args.list:
#     logger.info("Available datasets:")
#     for dataset_name in list_datasets():
#         info = get_dataset_info(dataset_name)
#         status = "✓" if is_dataset_downloaded(dataset_name) else "✗"
#         logger.info(f"  {status} {dataset_name}: {info['name']}")
#     return 0

# # Show status
# if args.status:
#     logger.info("Dataset Status:")
#     logger.info("="*60)
#     for dataset_name in list_datasets():
#         info = get_dataset_info(dataset_name)
#         stats = get_dataset_statistics(dataset_name)
        
#         logger.info(f"\n{dataset_name}:")
#         logger.info(f"  Name: {info['name']}")
#         logger.info(f"  Downloaded: {'Yes' if stats['exists'] else 'No'}")
#         if stats['exists']:
#             logger.info(f"  Files: {stats['num_files']}")
#             logger.info(f"  Size: {stats['size_mb']:.2f} MB")
#             logger.info(f"  Path: {stats['path']}")
#     logger.info("="*60)
#     return 0

# # Create downloader
# downloader = DatasetDownloader(verbose=args.verbose)

# # Determine datasets to download
# if args.dataset:
#     datasets_to_download = [args.dataset]
# else:
#     datasets_to_download = list_datasets()

# # Dry run
# if args.dry_run:
#     logger.info("Dry run mode - checking availability...")
#     for dataset_name in datasets_to_download:
#         info = get_dataset_info(dataset_name)
#         logger.info(f"{dataset_name}: {info['name']}")
#     return 0

# # Download datasets
# logger.info(f"Downloading {len(datasets_to_download)} dataset(s)...")

# for dataset_name in datasets_to_download:
#     downloader.download_dataset(
#         dataset_name=dataset_name,
#         sample_size=args.samples,
#         force=args.force
#     )

# # Print summary
# downloader.print_summary()

# return 0
# if name == 'main':
# sys.exit(main())

# text

# ***

# ## **Summary**

# ### **Files Implemented:**

# ✅ **`evaluation/datasets/__init__.py`** (220 lines)
# - Package initialization
# - Dataset registry
# - Path management
# - Status checking
# - Statistics computation
# - Validation
# - Cache management

# ✅ **`evaluation/datasets/download_datasets.py`** (530 lines)
# - Complete download orchestrator
# - FaceForensics++ instructions
# - Celeb-DF instructions
# - Synthetic data generation
# - Command-line interface
# - Progress tracking
# - Validation

# ### **Features:**

# ✅ **Dataset Package:**
# - Registry of all datasets
# - Path resolution
# - Download status checking
# - Statistics computation
# - Validation framework
# - Cache management

# ✅ **Download Script:**
# - Manual download instructions (FaceForensics++, Celeb-DF)
# - Automated synthetic data generation
# - Directory structure creation
# - README generation
# - Status reporting
# - Force re-download option

# ### **Key Capabilities:**

# 1. **FaceForensics++:**
#    - Instructions for manual download
#    - Directory structure creation
#    - README with citations
#    - Validation of downloaded files

# 2. **Celeb-DF:**
#    - Access request instructions
#    - Proper directory layout
#    - License information

# 3. **Synthetic Attacks:**
#    - Automated generation
#    - Configurable parameters
#    - JSON format output

# ### **Usage Examples:**

# ```bash
# # List available datasets
# python evaluation/datasets/download_datasets.py --list

# # Check download status
# python evaluation/datasets/download_datasets.py --status

# # Download all datasets
# python evaluation/datasets/download_datasets.py

# # Download specific dataset
# python evaluation/datasets/download_datasets.py --dataset synthetic_attacks

# # Generate synthetic attacks
# python evaluation/datasets/download_datasets.py --dataset synthetic_attacks --samples 20

# # Force re-download
# python evaluation/datasets/download_datasets.py --dataset faceforensics --force"