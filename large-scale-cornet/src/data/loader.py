# src/data/loader.py

"""
Data loader: download and prepare datasets
"""

import os
import numpy as np
from PIL import Image
from pathlib import Path


def check_dataset_exists(data_dir, n_people=40, imgs_per_person=10):
    """
    Check if dataset already exists and is valid.
    
    Parameters:
        data_dir: path to dataset directory
        n_people: expected number of people
        imgs_per_person: expected images per person
        
    Returns:
        bool: True if dataset exists and is valid
    """
    if not os.path.exists(data_dir):
        return False
    
    person_dirs = sorted([d for d in os.listdir(data_dir) 
                         if os.path.isdir(os.path.join(data_dir, d)) 
                         and not d.startswith('.')])
    
    if len(person_dirs) != n_people:
        return False
    
    # Check each person has correct number of images
    for person_name in person_dirs:
        person_dir = os.path.join(data_dir, person_name)
        images = [f for f in os.listdir(person_dir)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if len(images) != imgs_per_person:
            return False
    
    return True


def prepare_balanced_lfw(save_dir, n_people=40, imgs_per_person=10, 
                        local_lfw_dir=None, force_create=False):
    """
    Create a perfectly balanced LFW dataset subset.
    
    Can use local LFW data or download from sklearn.
    
    Parameters:
        save_dir: directory to save the balanced dataset
        n_people: number of people (classes) to include
        imgs_per_person: exact number of images per person
        local_lfw_dir: path to local LFW directory (optional)
                      If provided, use local data instead of downloading
                      e.g., '.../data/lfw/lfw-deepfunneled'
        force_create: if True, recreate even if data exists
        
    Returns:
        dict: statistics about the dataset
    """
    import shutil
    
    # Check if dataset already exists
    if not force_create and check_dataset_exists(save_dir, n_people, imgs_per_person):
        print(f"Dataset already exists: {save_dir}")
        print(f"Validation: {n_people} people x {imgs_per_person} images = {n_people * imgs_per_person} images")
        print("Use force_create=True to recreate.")
        
        return {
            'save_dir': save_dir,
            'n_people': n_people,
            'imgs_per_person': imgs_per_person,
            'total_images': n_people * imgs_per_person,
            'status': 'existing'
        }
    
    # Clean old directory if recreating
    if os.path.exists(save_dir):
        print(f"Removing existing directory: {save_dir}")
        shutil.rmtree(save_dir)
    
    os.makedirs(save_dir, exist_ok=True)
    
    # ====== Method 1: Use local LFW data ======
    if local_lfw_dir is not None:
        # Find LFW data directory (try common subdirectories)
        if not os.path.exists(local_lfw_dir):
            for subdir in ['lfw-deepfunneled/lfw-deepfunneled', 'lfw-deepfunneled', 'lfw']:
                test_path = os.path.join(local_lfw_dir, subdir)
                if os.path.exists(test_path):
                    local_lfw_dir = test_path
                    break
        
        if not os.path.exists(local_lfw_dir):
            print(f"Error: Local LFW directory not found: {local_lfw_dir}")
            return None
        
        print(f"Using local LFW data: {local_lfw_dir}")
        
        # Try to use CSV file first (much faster)
        csv_path = None
        parent_dir = os.path.dirname(local_lfw_dir)
        
        # Try to find lfw_allnames.csv
        possible_csv_paths = [
            os.path.join(parent_dir, 'lfw_allnames.csv'),
            os.path.join(os.path.dirname(parent_dir), 'lfw_allnames.csv'),
            os.path.join(local_lfw_dir, '..', 'lfw_allnames.csv')
        ]
        
        for path in possible_csv_paths:
            if os.path.exists(path):
                csv_path = path
                break
        
        person_image_counts = {}
        
        if csv_path:
            print(f"Using CSV file: {csv_path}")
            print(f"Loading people with >= {imgs_per_person} images...")
            
            try:
                import csv
                with open(csv_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    qualified_people = []
                    
                    for row in reader:
                        name = row['name']
                        num_images = int(row['images'])
                        
                        if num_images >= imgs_per_person:
                            qualified_people.append((name, num_images))
                    
                    print(f"Found {len(qualified_people)} people with >= {imgs_per_person} images from CSV")
                    
                    # Sort by image count and select top N
                    qualified_people.sort(key=lambda x: x[1], reverse=True)
                    selected_people = qualified_people[:n_people]
                    
                    # Store selected people (will verify folders during copying)
                    for name, count in selected_people:
                        person_image_counts[name] = (count, None)  # images list will be loaded during copying
                    
                    print(f"Selected {len(person_image_counts)} people from CSV")
                    
            except Exception as e:
                print(f"Warning: Failed to read CSV file: {e}")
                print("Falling back to directory scanning...")
                csv_path = None
        
        # Fallback: scan directories if CSV not available or failed
        if not csv_path or len(person_image_counts) == 0:
            print("Scanning directories for people with enough images...")
            
            person_dirs = [d for d in os.listdir(local_lfw_dir)
                          if os.path.isdir(os.path.join(local_lfw_dir, d))
                          and not d.startswith('.')]
            
            if len(person_dirs) == 0:
                print(f"Error: No person directories found in {local_lfw_dir}")
                return None
            
            print(f"Found {len(person_dirs)} person directories")
            
            # Scan all or enough directories
            max_scan = len(person_dirs) if imgs_per_person > 20 else min(len(person_dirs), n_people * 10)
            
            for idx, person_name in enumerate(person_dirs[:max_scan]):
                # Progress indicator every 500 people
                if (idx + 1) % 500 == 0:
                    print(f"  Scanned {idx + 1}/{max_scan} people, found {len(person_image_counts)} qualified")
                
                person_path = os.path.join(local_lfw_dir, person_name)
                
                try:
                    files = os.listdir(person_path)
                    image_count = sum(1 for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png')))
                    
                    if image_count >= imgs_per_person:
                        images = sorted([f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                        person_image_counts[person_name] = (image_count, images)
                        
                        # Early termination only after we have enough
                        if len(person_image_counts) >= n_people * 2:
                            print(f"  Found enough candidates ({len(person_image_counts)}), stopping scan")
                            break
                            
                except Exception as e:
                    continue
            
            print(f"Found {len(person_image_counts)} people with >= {imgs_per_person} images")
        
        # Check if we have enough people
        if len(person_image_counts) == 0:
            print(f"Error: No people found with >= {imgs_per_person} images")
            return None
        
        # Adjust n_people if we don't have enough (allow using all available)
        actual_n_people = min(n_people, len(person_image_counts))
        
        if actual_n_people < n_people:
            print(f"Note: Using {actual_n_people} people (requested {n_people}, but only {len(person_image_counts)} available)")
        
        # Select top N people by image count
        sorted_people = sorted(person_image_counts.items(), 
                              key=lambda x: x[1][0], 
                              reverse=True)[:actual_n_people]
        
        print(f"Creating balanced dataset: {actual_n_people} people x {imgs_per_person} images")
        
        total_saved = 0
        skipped_people = []
        
        for person_name, (count, images) in sorted_people:
            person_save_dir = os.path.join(save_dir, person_name)
            os.makedirs(person_save_dir, exist_ok=True)
            
            # If images list is None (from CSV), load it now
            if images is None:
                person_path = os.path.join(local_lfw_dir, person_name)
                
                # Verify folder exists
                if not os.path.exists(person_path):
                    print(f"Warning: Folder not found for {person_name}, skipping")
                    skipped_people.append(person_name)
                    os.rmdir(person_save_dir)  # Remove empty folder
                    continue
                
                try:
                    files = os.listdir(person_path)
                    images = sorted([f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                    
                    # Verify we have enough images
                    if len(images) < imgs_per_person:
                        print(f"Warning: {person_name} has only {len(images)} images (need {imgs_per_person}), skipping")
                        skipped_people.append(person_name)
                        os.rmdir(person_save_dir)  # Remove empty folder
                        continue
                        
                except Exception as e:
                    print(f"Warning: Failed to read folder for {person_name}: {e}, skipping")
                    skipped_people.append(person_name)
                    os.rmdir(person_save_dir)  # Remove empty folder
                    continue
            
            # Copy first N images
            person_saved = 0
            for i, img_name in enumerate(images[:imgs_per_person]):
                src_path = os.path.join(local_lfw_dir, person_name, img_name)
                dst_path = os.path.join(person_save_dir, f"{person_name}_{i:03d}.jpg")
                
                try:
                    img = Image.open(src_path).convert('RGB')
                    img.save(dst_path, quality=95)
                    total_saved += 1
                    person_saved += 1
                except Exception as e:
                    print(f"Warning: Failed to process {src_path}: {e}")
            
            # If we couldn't save enough images, remove the person folder
            if person_saved < imgs_per_person:
                print(f"Warning: Could only save {person_saved}/{imgs_per_person} images for {person_name}, removing")
                shutil.rmtree(person_save_dir)
                skipped_people.append(person_name)
        
        if skipped_people:
            print(f"\nSkipped {len(skipped_people)} people due to missing/invalid data")
            final_people_count = actual_n_people - len(skipped_people)
            print(f"Successfully created dataset with {final_people_count} people")
        else:
            final_people_count = actual_n_people
    
    # ====== Method 2: Download from sklearn ======
    else:
        try:
            from sklearn.datasets import fetch_lfw_people
        except ImportError:
            print("Error: sklearn required for downloading")
            print("Install: pip install scikit-learn")
            return None
        
        print(f"Downloading LFW dataset...")
        print(f"Creating: {n_people} people x {imgs_per_person} images")
        
        # Download LFW
        lfw = fetch_lfw_people(min_faces_per_person=imgs_per_person, resize=None, color=True)
        
        # Select top N people by image count
        unique_ids, counts = np.unique(lfw.target, return_counts=True)
        sorted_indices = np.argsort(counts)[::-1][:n_people]
        selected_person_ids = unique_ids[sorted_indices]
        selected_counts = counts[sorted_indices]
        
        print(f"Selected {len(selected_person_ids)} people")
        print(f"Image count range: {selected_counts.min()}-{selected_counts.max()}")
        
        if selected_counts.min() < imgs_per_person:
            print(f"Error: Some people have < {imgs_per_person} images")
            return None
        
        total_saved = 0
        
        # Process each person
        for person_id in selected_person_ids:
            person_name = lfw.target_names[person_id].replace(" ", "_")
            indices = np.where(lfw.target == person_id)[0][:imgs_per_person]
            
            person_dir = os.path.join(save_dir, person_name)
            os.makedirs(person_dir, exist_ok=True)
            
            for i, img_idx in enumerate(indices):
                img_array = lfw.images[img_idx]
                
                # Convert to uint8
                if img_array.max() <= 1.0:
                    img_array = (img_array * 255).astype(np.uint8)
                else:
                    img_array = img_array.astype(np.uint8)
                
                # Handle grayscale or color
                if len(img_array.shape) == 2:
                    im = Image.fromarray(img_array, mode='L').convert('RGB')
                else:
                    im = Image.fromarray(img_array, mode='RGB')
                
                save_path = os.path.join(person_dir, f"{person_name}_{i:03d}.jpg")
                im.save(save_path, quality=95)
                total_saved += 1
    
    # Calculate actual number of people created
    if local_lfw_dir is not None:
        # For local LFW, count the actual directories created
        if os.path.exists(save_dir):
            created_people = len([d for d in os.listdir(save_dir) 
                                 if os.path.isdir(os.path.join(save_dir, d)) 
                                 and not d.startswith('.')])
        else:
            created_people = 0
    else:
        # For sklearn download, use the calculated n_people
        created_people = n_people
    
    print(f"\nComplete: {save_dir}")
    print(f"Created: {created_people} people x {imgs_per_person} images = {total_saved} images")
    
    if created_people < n_people:
        print(f"Note: Requested {n_people} people, but only {created_people} were available/valid")
    
    return {
        'save_dir': save_dir,
        'n_people': created_people,
        'imgs_per_person': imgs_per_person,
        'total_images': total_saved,
        'status': 'created' if local_lfw_dir else 'downloaded'
    }


def validate_dataset(data_dir, verbose=True):
    """
    Validate dataset structure and report statistics.
    
    This function performs thorough validation:
    - Checks if directory exists
    - Counts people and images
    - Verifies all images can be opened
    - Reports any issues found
    
    Parameters:
        data_dir: path to dataset directory
        verbose: if True, print detailed validation report
        
    Returns:
        dict: statistics about the dataset
    """
    if not os.path.exists(data_dir):
        print(f"Error: Directory does not exist: {data_dir}")
        return None
    
    person_dirs = sorted([d for d in os.listdir(data_dir) 
                         if os.path.isdir(os.path.join(data_dir, d)) 
                         and not d.startswith('.')])
    
    if len(person_dirs) == 0:
        print("Error: No person directories found")
        return None
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Validating Dataset: {data_dir}")
        print(f"{'='*60}")
    
    image_counts = []
    corrupted_images = []
    missing_folders = []
    
    for person_name in person_dirs:
        person_dir = os.path.join(data_dir, person_name)
        
        # Check if folder exists and is readable
        if not os.path.isdir(person_dir):
            missing_folders.append(person_name)
            continue
        
        try:
            images = [f for f in os.listdir(person_dir)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            image_counts.append(len(images))
            
            # Verify images can be opened (sample check for large datasets)
            if verbose and len(images) > 0:
                # Check first image of each person
                test_img_path = os.path.join(person_dir, images[0])
                try:
                    img = Image.open(test_img_path)
                    img.verify()  # Verify it's a valid image
                except Exception as e:
                    corrupted_images.append((person_name, images[0], str(e)))
                    
        except Exception as e:
            if verbose:
                print(f"Warning: Error reading {person_name}: {e}")
            image_counts.append(0)
    
    total_images = sum(image_counts)
    is_balanced = len(set(image_counts)) == 1
    
    if verbose:
        print(f"\n{'Basic Statistics:'}")
        print(f"  Total People: {len(person_dirs)}")
        print(f"  Total Images: {total_images}")
        print(f"  Images per Person: {image_counts[0] if is_balanced else f'{min(image_counts)}-{max(image_counts)}'}")
        print(f"  Dataset Balance: {'✓ PERFECT' if is_balanced else '✗ UNBALANCED'}")
        
        if is_balanced:
            print(f"  Expected Total: {len(person_dirs)} × {image_counts[0]} = {len(person_dirs) * image_counts[0]}")
        
        # Report issues
        issues_found = False
        
        if missing_folders:
            issues_found = True
            print(f"\n{'⚠ Issues Found:'}")
            print(f"  Missing/Invalid Folders: {len(missing_folders)}")
            for folder in missing_folders[:5]:  # Show first 5
                print(f"    - {folder}")
            if len(missing_folders) > 5:
                print(f"    ... and {len(missing_folders) - 5} more")
        
        if corrupted_images:
            if not issues_found:
                print(f"\n{'⚠ Issues Found:'}")
            issues_found = True
            print(f"  Corrupted Images: {len(corrupted_images)}")
            for person, img, error in corrupted_images[:3]:  # Show first 3
                print(f"    - {person}/{img}: {error}")
            if len(corrupted_images) > 3:
                print(f"    ... and {len(corrupted_images) - 3} more")
        
        if not is_balanced:
            if not issues_found:
                print(f"\n{'⚠ Issues Found:'}")
            issues_found = True
            print(f"  Unbalanced Dataset:")
            print(f"    Min images: {min(image_counts)}")
            print(f"    Max images: {max(image_counts)}")
            print(f"    Variation: {max(image_counts) - min(image_counts)}")
        
        if not issues_found:
            print(f"\n✓ Validation Passed: No issues found")
        
        print(f"{'='*60}\n")
    
    return {
        'num_people': len(person_dirs),
        'total_images': total_images,
        'is_balanced': is_balanced,
        'images_per_person': image_counts,
        'corrupted_images': corrupted_images,
        'missing_folders': missing_folders
    }

