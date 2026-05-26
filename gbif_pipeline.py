#!/usr/bin/env python3
"""
GBIF Image Downloader Pipeline for Great Salt Lake Wetland Species
Authored by Antigravity

This script queries the Global Biodiversity Information Facility (GBIF) Occurrence API
to gather high-quality field images of invasive phragmites and similar wetland plants
around the Great Salt Lake (GSL), Utah, or globally as fallback.

Key Features:
- Direct API interface to GBIF (no heavy external SDK needed).
- Multi-stage geographic expansion (GSL -> Utah -> Global) for rare species.
- Incremental downloading (resumes seamlessly and avoids duplicating existing images).
- Clean, multi-threaded image downloading with retries and PIL image validation (converts to RGB JPEG).
- Automatic database recording in dataset/metadata.csv with scientific citations and CC licenses.
- Automatic generation of a stunning interactive Leaflet.js HTML report dashboard.
"""

import os
import sys
import csv
import time
import argparse
import tempfile
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from PIL import Image

# Taxon Configurations using Scientific Names (resolved dynamically at runtime)
SPECIES_CONFIG = {
    "invasive_phragmites": {
        "display_name": "Invasive Phragmites (Phragmites australis)",
        "taxa": ["Phragmites australis"],
        "exclude_taxa": ["Phragmites australis subsp. americanus"]
    },
    "native_phragmites": {
        "display_name": "Native Phragmites (Phragmites australis subsp. americanus)",
        "taxa": ["Phragmites australis subsp. americanus"]
    },
    "cattails": {
        "display_name": "Cattails (Typha genus)",
        "taxa": ["Typha"]
    },
    "bulrushes": {
        "display_name": "Bulrushes (Schoenoplectus, Bolboschoenus, Scirpus)",
        "taxa": ["Schoenoplectus", "Bolboschoenus", "Scirpus"]
    },
    "pickleweed": {
        "display_name": "Pickleweed (Salicornia, Sarcocornia)",
        "taxa": ["Salicornia", "Sarcocornia"]
    },
    "saltgrass": {
        "display_name": "Saltgrass (Distichlis spicata)",
        "taxa": ["Distichlis spicata"]
    }
}

# Bounding Box Configurations (latitude, longitude ranges)
GEOGRAPHIC_STAGES = [
    {
        "name": "Great Salt Lake",
        "params": {
            "decimalLatitude": "40.2,41.9",
            "decimalLongitude": "-114.3,-111.7"
        }
    },
    {
        "name": "State of Utah",
        "params": {
            "decimalLatitude": "37.0,42.0",
            "decimalLongitude": "-114.0,-109.0"
        }
    },
    {
        "name": "Global (Unrestricted)",
        "params": {}  # No geographic restrictions
    }
]

# Headers for HTTP requests to be polite
HEADERS = {
    "User-Agent": "AntigravityWetlandPipeline/1.0 (schie@Desktop; contact: google-deepmind-antigravity)"
}

def setup_directories(output_dir):
    """Create the dataset directory and subdirectories for each class."""
    os.makedirs(output_dir, exist_ok=True)
    for class_name in SPECIES_CONFIG.keys():
        os.makedirs(os.path.join(output_dir, class_name), exist_ok=True)

def resolve_taxon_keys(species_config):
    """
    Queries the GBIF Species Match API for all scientific names listed in the config
    and populates matching numeric taxon keys dynamically at runtime.
    """
    resolved_config = {}
    
    print("\n[*] Resolving species names to GBIF taxon keys dynamically...")
    for class_name, config in species_config.items():
        resolved_keys = []
        for name in config["taxa"]:
            try:
                url = "https://api.gbif.org/v1/species/match"
                params = {"name": name, "strict": "true", "kingdom": "Plantae"}
                response = requests.get(url, params=params, headers=HEADERS, timeout=15)
                if response.status_code == 200:
                    data = response.json()
                    key = data.get("usageKey")
                    sc_name = data.get("scientificName", name)
                    if key:
                        resolved_keys.append(key)
                        print(f"    - Class: {class_name} | '{name}' -> '{sc_name}' (Taxon Key: {key})")
                    else:
                        print(f"    [!] Warning: Could not resolve usageKey for '{name}'")
                else:
                    print(f"    [!] Error: GBIF Match API returned status {response.status_code} for '{name}'")
            except Exception as e:
                print(f"    [!] Error resolving '{name}': {e}")
                
        # Resolve exclusion keys
        exclude_keys = []
        for name in config.get("exclude_taxa", []):
            try:
                url = "https://api.gbif.org/v1/species/match"
                params = {"name": name, "strict": "true", "kingdom": "Plantae"}
                response = requests.get(url, params=params, headers=HEADERS, timeout=15)
                if response.status_code == 200:
                    data = response.json()
                    key = data.get("usageKey")
                    if key:
                        exclude_keys.append(key)
                        print(f"    - Exclusion | '{name}' -> Taxon Key: {key}")
            except Exception as e:
                print(f"    [!] Error resolving exclusion '{name}': {e}")
                
        resolved_config[class_name] = {
            "display_name": config["display_name"],
            "taxon_keys": resolved_keys,
            "exclude_subspecies": exclude_keys
        }
    return resolved_config


def load_existing_metadata(metadata_path):
    """Load existing metadata and return a set of already downloaded GBIF occurrence IDs."""
    existing_ids = set()
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("gbif_id"):
                        existing_ids.add(row["gbif_id"])
            print(f"[*] Found {len(existing_ids)} existing occurrences in metadata CSV. Resuming incrementally...")
        except Exception as e:
            print(f"[!] Error reading metadata CSV ({e}). Starting fresh.")
    return existing_ids

def query_gbif_occurrences(class_name, config, target_count, existing_ids):
    """
    Queries the GBIF API in stages (GSL -> Utah -> Global) until target_count is reached
    or all available occurrences are exhausted.
    """
    occurrences_to_download = []
    already_seen_ids = set(existing_ids)  # Don't add duplicates in this run
    
    print(f"\n==================================================")
    print(f"Querying occurrences for: {class_name.upper()}")
    print(f"==================================================")

    for stage in GEOGRAPHIC_STAGES:
        needed = target_count - len(occurrences_to_download)
        if needed <= 0:
            break
        
        print(f"[*] Stage: {stage['name']} | Seeking {needed} more images...")
        
        # We will loop through the taxon keys associated with this class
        for taxon_key in config["taxon_keys"]:
            needed = target_count - len(occurrences_to_download)
            if needed <= 0:
                break
            
            offset = 0
            limit = 100
            
            while len(occurrences_to_download) < target_count:
                params = {
                    "taxonKey": taxon_key,
                    "mediaType": "StillImage",
                    "hasCoordinate": "true",
                    "hasGeospatialIssue": "false",
                    "basisOfRecord": "HUMAN_OBSERVATION",
                    "limit": limit,
                    "offset": offset
                }
                # Add geographic bounding box parameters if present
                params.update(stage["params"])
                
                try:
                    url = "https://api.gbif.org/v1/occurrence/search"
                    response = requests.get(url, params=params, headers=HEADERS, timeout=15)
                    if response.status_code != 200:
                        print(f"    [!] GBIF API returned error {response.status_code}")
                        break
                    
                    data = response.json()
                    results = data.get("results", [])
                    total_count = data.get("count", 0)
                    
                    if not results:
                        break
                        
                    for rec in results:
                        gbif_id = str(rec.get("key"))
                        
                        # Apply subspecies exclusion filters
                        exclude_keys = config.get("exclude_subspecies", [])
                        sub_key = rec.get("subspeciesKey")
                        tax_key = rec.get("taxonKey")
                        if sub_key in exclude_keys or tax_key in exclude_keys:
                            continue
                        
                        # Double check that we don't already have this occurrence ID
                        if gbif_id in already_seen_ids:
                            continue
                            
                        # Extract media information
                        media = rec.get("media", [])
                        image_url = None
                        for m in media:
                            if m.get("type") == "StillImage" and m.get("identifier"):
                                temp_url = m["identifier"]
                                # Avoid herbarium specimen sheets to ensure high quality field photos for training
                                bad_domains = ["sweetgum.nybg.org", "idigbio.org", "herbarium", "si.edu", "jacq.org", "unibio.unam.mx", "swbiodiversity.org"]
                                if any(bd in temp_url.lower() for bd in bad_domains):
                                    continue
                                image_url = temp_url
                                break
                                
                        if image_url:
                            occurrences_to_download.append({
                                "gbif_id": gbif_id,
                                "class": class_name,
                                "scientific_name": rec.get("scientificName", "Unknown"),
                                "latitude": rec.get("decimalLatitude"),
                                "longitude": rec.get("decimalLongitude"),
                                "state_province": rec.get("stateProvince", ""),
                                "country": rec.get("country", ""),
                                "event_date": rec.get("eventDate", ""),
                                "license": rec.get("license", "Unknown"),
                                "publisher": rec.get("publisher", ""),
                                "rights_holder": rec.get("rightsHolder", ""),
                                "image_url": image_url
                            })
                            already_seen_ids.add(gbif_id)
                            
                        if len(occurrences_to_download) >= target_count:
                            break
                            
                    offset += limit
                    if offset >= total_count or offset >= 5000: # cap pagination to be safe
                        break
                        
                    time.sleep(0.1)  # Polite sleep between pages
                    
                except Exception as e:
                    print(f"    [!] Error querying GBIF API: {e}")
                    break
                    
    print(f"[*] Found {len(occurrences_to_download)} new candidate occurrences to download for {class_name}.")
    return occurrences_to_download

def download_single_image(occ, output_dir, session):
    """
    Downloads a single image from URL, validates that it's a correct image,
    converts it to a standard RGB JPEG, and returns metadata if successful.
    """
    gbif_id = occ["gbif_id"]
    class_name = occ["class"]
    image_url = occ["image_url"]
    
    # Target image filename and path
    relative_path = f"dataset/{class_name}/{gbif_id}.jpg"
    final_path = os.path.join(output_dir, "..", relative_path)
    final_path = os.path.normpath(final_path)
    
    # Try downloading with standard retries
    retries = 2
    for attempt in range(retries + 1):
        try:
            # Download to a temporary file first
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_path = tmp_file.name
                
            response = session.get(image_url, headers=HEADERS, timeout=12, stream=True)
            if response.status_code != 200:
                os.remove(tmp_path)
                continue
                
            with open(tmp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            # Check file size (discard if < 2KB, likely broken/error page)
            if os.path.getsize(tmp_path) < 2048:
                os.remove(tmp_path)
                continue
                
            # Verify and convert using PIL
            with Image.open(tmp_path) as img:
                img.verify()  # verify image structure
                
            with Image.open(tmp_path) as img:
                rgb_img = img.convert("RGB")
                rgb_img.save(final_path, "JPEG", quality=90)
                
            os.remove(tmp_path)
            
            # Update occurrence with final path and return
            occ["image_path"] = relative_path.replace("\\", "/")
            return occ
            
        except Exception as e:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except:
                    pass
            if attempt == retries:
                # Failed all attempts
                return None
            time.sleep(0.5)
            
    return None

def download_images_parallel(occurrences, output_dir, num_workers):
    """Downloads candidate occurrences in parallel using a ThreadPoolExecutor."""
    if not occurrences:
        return []
        
    print(f"[*] Starting download of {len(occurrences)} images with {num_workers} parallel workers...")
    
    downloaded_records = []
    session = requests.Session()
    
    # Limit connections pool size in the session to match num_workers
    adapter = requests.adapters.HTTPAdapter(pool_connections=num_workers, pool_maxsize=num_workers)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    completed = 0
    total = len(occurrences)
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(download_single_image, occ, output_dir, session): occ for occ in occurrences}
        
        for future in as_completed(futures):
            occ = futures[future]
            completed += 1
            try:
                record = future.result()
                if record:
                    downloaded_records.append(record)
                    print(f"    [{completed}/{total}] SUCCESS: Downloaded {record['class']} (ID: {record['gbif_id']})")
                else:
                    print(f"    [{completed}/{total}] FAILED: {occ['class']} from URL {occ['image_url'][:50]}...")
            except Exception as e:
                print(f"    [{completed}/{total}] EXCEPTION: {occ['class']} ({e})")
                
    return downloaded_records

def write_metadata_csv(metadata_path, downloaded_records):
    """Appends downloaded records to the metadata CSV file."""
    if not downloaded_records:
        return
        
    file_exists = os.path.exists(metadata_path)
    
    fields = [
        "image_path", "gbif_id", "class", "scientific_name", 
        "latitude", "longitude", "state_province", "country", 
        "event_date", "license", "publisher", "rights_holder", "image_url"
    ]
    
    try:
        with open(metadata_path, mode="a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if not file_exists:
                writer.writeheader()
            for record in downloaded_records:
                # filter out non-field keys just in case
                row = {k: record[k] for k in fields if k in record}
                writer.writerow(row)
        print(f"[*] Metadata successfully written to: {metadata_path}")
    except Exception as e:
        print(f"[!] Error writing metadata CSV: {e}")

def generate_interactive_report(metadata_path, report_path):
    """Reads the metadata CSV and creates an interactive HTML dashboard with Leaflet.js."""
    if not os.path.exists(metadata_path):
        print("[!] No metadata CSV found. Cannot generate report.")
        return
        
    records = []
    class_counts = {k: 0 for k in SPECIES_CONFIG.keys()}
    license_counts = {}
    
    try:
        with open(metadata_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(row)
                c = row.get("class")
                if c in class_counts:
                    class_counts[c] += 1
                lic = row.get("license", "Unknown")
                # simplify license string for charts
                if "cc0" in lic.lower() or "publicdomain" in lic.lower():
                    lic_short = "CC0 / Public Domain"
                elif "by-nc" in lic.lower():
                    lic_short = "CC BY-NC"
                elif "by" in lic.lower():
                    lic_short = "CC BY"
                else:
                    lic_short = "Other / Restricted"
                license_counts[lic_short] = license_counts.get(lic_short, 0) + 1
    except Exception as e:
        print(f"[!] Error reading metadata for HTML report: {e}")
        return
        
    # Build JS-safe JSON records for mapping (only those with lat/lon)
    map_pins = []
    for r in records:
        try:
            if r.get("latitude") and r.get("longitude"):
                map_pins.append({
                    "id": r["gbif_id"],
                    "class": r["class"],
                    "name": r["scientific_name"].split(" ")[0] + " " + r["scientific_name"].split(" ")[1] if len(r["scientific_name"].split(" ")) > 1 else r["scientific_name"],
                    "lat": float(r["latitude"]),
                    "lon": float(r["longitude"]),
                    "path": r["image_path"],
                    "license": r["license"].split("/")[-1].upper() if "/" in r["license"] else "CC BY"
                })
        except:
            continue
            
    # Sample cards (up to 12 for the dashboard gallery)
    gallery_items = records[:30] # take up to 30 items
    
    # HTML Content placeholder definition to satisfy f-string
    json_pins_placeholder = "{json_pins_placeholder}"
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Great Salt Lake Wetland Species - Dataset Dashboard</title>
    
    <!-- Google Fonts & Leaflet Map -->
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    
    <style>
        :root {{
            --bg-color: #0b0f19;
            --card-bg: #141d2f;
            --text-color: #f3f4f6;
            --text-muted: #9ca3af;
            --accent-primary: #10b981; /* Emerald */
            --accent-sec: #06b6d4; /* Cyan */
            --border-color: #1e293b;
            
            /* Class Colors */
            --color-invasive: #ef4444; /* Red */
            --color-native: #10b981; /* Emerald */
            --color-cattails: #f59e0b; /* Amber */
            --color-bulrushes: #8b5cf6; /* Purple */
            --color-pickleweed: #ec4899; /* Pink */
            --color-saltgrass: #3b82f6; /* Blue */
        }}
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            background-color: var(--bg-color);
            color: var(--text-color);
            font-family: 'Inter', sans-serif;
            line-height: 1.5;
            padding-bottom: 5rem;
        }}
        
        header {{
            background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%);
            border-bottom: 1px solid var(--border-color);
            padding: 2.5rem 2rem;
            text-align: center;
            position: relative;
            overflow: hidden;
        }}
        
        header::after {{
            content: '';
            position: absolute;
            bottom: 0;
            left: 0;
            width: 100%;
            height: 3px;
            background: linear-gradient(90deg, var(--accent-primary), var(--accent-sec));
        }}
        
        h1 {{
            font-family: 'Outfit', sans-serif;
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            letter-spacing: -0.025em;
        }}
        
        .subtitle {{
            color: var(--text-muted);
            font-size: 1.1rem;
            font-weight: 300;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
        }}
        
        /* Stats Dashboard */
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }}
        
        .stat-card {{
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
            transition: all 0.3s ease;
        }}
        
        .stat-card:hover {{
            transform: translateY(-2px);
            border-color: #334155;
            box-shadow: 0 10px 20px rgba(0,0,0,0.3);
        }}
        
        .stat-val {{
            font-size: 2rem;
            font-weight: 700;
            font-family: 'Outfit', sans-serif;
            margin-bottom: 0.25rem;
        }}
        
        .stat-lbl {{
            color: var(--text-muted);
            font-size: 0.875rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        
        /* Grid Layout */
        .dashboard-grid {{
            display: grid;
            grid-template-columns: 3fr 2fr;
            gap: 2rem;
            margin-bottom: 2.5rem;
        }}
        
        @media (max-width: 1024px) {{
            .dashboard-grid {{
                grid-template-columns: 1fr;
            }}
        }}
        
        .section-card {{
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 1.5rem;
            overflow: hidden;
        }}
        
        .section-title {{
            font-family: 'Outfit', sans-serif;
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: 1.25rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}
        
        /* Map styling */
        #map {{
            height: 480px;
            width: 100%;
            border-radius: 10px;
            border: 1px solid var(--border-color);
            z-index: 1;
        }}
        
        /* Class List styling */
        .class-list {{
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }}
        
        .class-item {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            background-color: rgba(255,255,255,0.02);
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid var(--border-color);
        }}
        
        .class-item.invasive_phragmites {{ border-left-color: var(--color-invasive); }}
        .class-item.native_phragmites {{ border-left-color: var(--color-native); }}
        .class-item.cattails {{ border-left-color: var(--color-cattails); }}
        .class-item.bulrushes {{ border-left-color: var(--color-bulrushes); }}
        .class-item.pickleweed {{ border-left-color: var(--color-pickleweed); }}
        .class-item.saltgrass {{ border-left-color: var(--color-saltgrass); }}
        
        .class-details {{
            display: flex;
            flex-direction: column;
        }}
        
        .class-name {{
            font-weight: 600;
            font-size: 0.95rem;
        }}
        
        .class-qty {{
            font-family: 'Outfit', sans-serif;
            font-size: 1.25rem;
            font-weight: 700;
        }}
        
        /* Gallery */
        .gallery-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
            gap: 1.5rem;
            margin-top: 1rem;
        }}
        
        .gallery-card {{
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            overflow: hidden;
            transition: all 0.3s ease;
        }}
        
        .gallery-card:hover {{
            transform: scale(1.02);
            border-color: #475569;
        }}
        
        .gallery-img-container {{
            position: relative;
            height: 160px;
            width: 100%;
            background-color: #000;
            overflow: hidden;
        }}
        
        .gallery-img {{
            width: 100%;
            height: 100%;
            object-fit: cover;
        }}
        
        .gallery-badge {{
            position: absolute;
            top: 10px;
            left: 10px;
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-size: 0.7rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #fff;
        }}
        
        .badge-invasive_phragmites {{ background-color: var(--color-invasive); }}
        .badge-native_phragmites {{ background-color: var(--color-native); }}
        .badge-cattails {{ background-color: var(--color-cattails); }}
        .badge-bulrushes {{ background-color: var(--color-bulrushes); }}
        .badge-pickleweed {{ background-color: var(--color-pickleweed); }}
        .badge-saltgrass {{ background-color: var(--color-saltgrass); }}
        
        .gallery-info {{
            padding: 1rem;
        }}
        
        .gallery-title {{
            font-size: 0.85rem;
            font-weight: 600;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            font-style: italic;
        }}
        
        .gallery-meta {{
            color: var(--text-muted);
            font-size: 0.75rem;
            margin-top: 0.25rem;
            display: flex;
            justify-content: space-between;
        }}
        
        /* Leaflet popup customization */
        .leaflet-popup-content-wrapper {{
            background-color: var(--card-bg) !important;
            color: var(--text-color) !important;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            font-family: 'Inter', sans-serif;
        }}
        
        .leaflet-popup-tip {{
            background-color: var(--card-bg) !important;
        }}
        
        .popup-card {{
            text-align: center;
            max-width: 180px;
        }}
        
        .popup-img {{
            width: 100%;
            height: 90px;
            object-fit: cover;
            border-radius: 4px;
            margin-top: 5px;
        }}
        
        .popup-title {{
            font-size: 0.85rem;
            font-weight: 600;
            margin-top: 5px;
            font-style: italic;
        }}
        
        .popup-badge {{
            display: inline-block;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 0.65rem;
            font-weight: 700;
            color: #fff;
            margin-bottom: 3px;
        }}
    </style>
</head>
<body>

    <header>
        <h1>Wetland Plant Species Dataset</h1>
        <div class="subtitle">Great Salt Lake Machine Learning Pipeline | Image Gatherer Dashboard</div>
    </header>

    <div class="container">
        
        <!-- Summary Stats Row -->
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-val" style="color: var(--accent-primary);">{len(records)}</div>
                <div class="stat-lbl">Total Images</div>
            </div>
            <div class="stat-card">
                <div class="stat-val" style="color: var(--accent-sec);">{len(map_pins)}</div>
                <div class="stat-lbl">Mapped Georeferences</div>
            </div>
            <div class="stat-card">
                <div class="stat-val" style="color: #a855f7;">{len(license_counts)}</div>
                <div class="stat-lbl">CC License Variants</div>
            </div>
            <div class="stat-card">
                <div class="stat-val" style="color: #f59e0b;">6</div>
                <div class="stat-lbl">Target Classes</div>
            </div>
        </div>
        
        <!-- Interactive Map & Class Summary -->
        <div class="dashboard-grid">
            
            <!-- Map Container -->
            <div class="section-card">
                <div class="section-title">
                    <span>Geographic Distribution</span>
                    <span style="font-size: 0.8rem; color: var(--text-muted); font-weight: normal;">*Pins colored by species category</span>
                </div>
                <div id="map"></div>
            </div>
            
            <!-- Class Counts Card -->
            <div class="section-card">
                <div class="section-title">Class Distribution</div>
                <div class="class-list">
                    <div class="class-item invasive_phragmites">
                        <div class="class-details">
                            <span class="class-name">Invasive Phragmites</span>
                            <span style="font-size: 0.75rem; color: var(--text-muted);">P. australis subsp. australis</span>
                        </div>
                        <div class="class-qty">{class_counts.get("invasive_phragmites", 0)}</div>
                    </div>
                    <div class="class-item native_phragmites">
                        <div class="class-details">
                            <span class="class-name">Native Phragmites</span>
                            <span style="font-size: 0.75rem; color: var(--text-muted);">P. australis subsp. americanus</span>
                        </div>
                        <div class="class-qty">{class_counts.get("native_phragmites", 0)}</div>
                    </div>
                    <div class="class-item cattails">
                        <div class="class-details">
                            <span class="class-name">Cattails</span>
                            <span style="font-size: 0.75rem; color: var(--text-muted);">Typha spp.</span>
                        </div>
                        <div class="class-qty">{class_counts.get("cattails", 0)}</div>
                    </div>
                    <div class="class-item bulrushes">
                        <div class="class-details">
                            <span class="class-name">Bulrushes</span>
                            <span style="font-size: 0.75rem; color: var(--text-muted);">Schoenoplectus / Bolboschoenus</span>
                        </div>
                        <div class="class-qty">{class_counts.get("bulrushes", 0)}</div>
                    </div>
                    <div class="class-item pickleweed">
                        <div class="class-details">
                            <span class="class-name">Pickleweed</span>
                            <span style="font-size: 0.75rem; color: var(--text-muted);">Salicornia / Sarcocornia</span>
                        </div>
                        <div class="class-qty">{class_counts.get("pickleweed", 0)}</div>
                    </div>
                    <div class="class-item saltgrass">
                        <div class="class-details">
                            <span class="class-name">Saltgrass</span>
                            <span style="font-size: 0.75rem; color: var(--text-muted);">Distichlis spicata</span>
                        </div>
                        <div class="class-qty">{class_counts.get("saltgrass", 0)}</div>
                    </div>
                </div>
            </div>
            
        </div>
        
        <!-- Image Validation Gallery -->
        <div class="section-card">
            <div class="section-title">Recent Downloads Gallery</div>
            <div class="gallery-grid">
                """
                
    for item in gallery_items:
        class_lbl = item["class"].replace("_", " ").title()
        html_content += f"""
                <div class="gallery-card">
                    <div class="gallery-img-container">
                        <img class="gallery-img" src="{item['image_path']}" alt="{item['scientific_name']}" onerror="this.src='https://placehold.co/300x200?text=Image+Load+Error'">
                        <span class="gallery-badge badge-{item['class']}">{class_lbl}</span>
                    </div>
                    <div class="gallery-info">
                        <div class="gallery-title" title="{item['scientific_name']}">{item['scientific_name']}</div>
                        <div class="gallery-meta">
                            <span>ID: {item['gbif_id']}</span>
                            <span style="font-size: 0.7rem; color: var(--accent-sec);">{item['license'].split('/')[-1].upper() if '/' in item['license'] else 'CC'}</span>
                        </div>
                    </div>
                </div>
        """
        
    html_content += f"""
            </div>
        </div>
        
    </div>

    <!-- Leaflet Script for interactive map -->
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script>
        // Init Map centering on the Great Salt Lake
        const map = L.map('map').setView([41.1, -112.5], 8);
        
        // Dark theme map tiles
        L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
            subdomains: 'abcd',
            maxZoom: 20
        }}).addTo(map);
        
        // Colors mapping
        const classColors = {{
            "invasive_phragmites": "#ef4444",
            "native_phragmites": "#10b981",
            "cattails": "#f59e0b",
            "bulrushes": "#8b5cf6",
            "pickleweed": "#ec4899",
            "saltgrass": "#3b82f6"
        }};
        
        const mapPins = {json_pins_placeholder};
        
        // Add markers
        mapPins.forEach(pin => {{
            const markerColor = classColors[pin.class] || '#ffffff';
            
            // Simple circular marker
            const marker = L.circleMarker([pin.lat, pin.lon], {{
                radius: 6,
                fillColor: markerColor,
                color: '#ffffff',
                weight: 1,
                opacity: 0.8,
                fillOpacity: 0.7
            }}).addTo(map);
            
            // Popup contents
            const classTitle = pin.class.replace(/_/g, ' ').toUpperCase();
            const popupContent = `
                <div class="popup-card">
                    <span class="popup-badge" style="background-color: ${{markerColor}}">${{classTitle}}</span>
                    <div class="popup-title">${{pin.name}}</div>
                    <img class="popup-img" src="${{pin.path}}" onerror="this.src='https://placehold.co/150x100?text=Load+Failed'">
                    <div style="font-size: 0.65rem; color: #9ca3af; margin-top: 4px;">License: ${{pin.license}}</div>
                </div>
            `;
            marker.bindPopup(popupContent);
        }});
        
        // If pins exist, fit bounds
        if(mapPins.length > 0) {{
            const group = new L.featureGroup(mapPins.map(p => L.marker([p.lat, p.lon])));
            map.fitBounds(group.getBounds().pad(0.1));
        }}
    </script>
</body>
</html>
"""
    # Replace placeholder with actual json
    import json
    html_content = html_content.replace("{json_pins_placeholder}", json.dumps(map_pins))
    
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"[*] Interactive dataset dashboard generated at: {report_path}")
    except Exception as e:
        print(f"[!] Error writing HTML report: {e}")

def main():
    parser = argparse.ArgumentParser(description="GBIF Wetland Plants Image Pipeline")
    parser.add_argument("--target-count", type=int, default=200, help="Target image count per species class (default: 200)")
    parser.add_argument("--workers", type=int, default=8, help="Number of concurrent download worker threads (default: 8)")
    parser.add_argument("--output-dir", type=str, default="dataset", help="Output directory for downloaded images")
    parser.add_argument("--report", type=str, default="dataset_report.html", help="Path to write the interactive HTML report dashboard")
    parser.add_argument("--dry-run", action="store_true", help="Search occurrences and count candidates but do not download")
    args = parser.parse_args()

    print("[*] Starting GBIF Great Salt Lake Wetland Species Downloader Pipeline...")
    print(f"[*] Configured Parameters:")
    print(f"    - Target Count per Class: {args.target_count}")
    print(f"    - Concurrent Workers: {args.workers}")
    print(f"    - Output Directory: {args.output_dir}")
    print(f"    - Report Output Path: {args.report}")
    print(f"    - Dry Run: {args.dry_run}")
    
    setup_directories(args.output_dir)
    metadata_path = os.path.join(args.output_dir, "metadata.csv")
    
    # Resolve taxon keys dynamically via GBIF Species Match API
    resolved_config = resolve_taxon_keys(SPECIES_CONFIG)
    
    # Load existing downloaded GBIF IDs (enables incremental resume!)
    existing_ids = load_existing_metadata(metadata_path)
    
    all_candidates = []
    
    # Stage 1: Query GBIF to discover candidates we need to download
    for class_name, config in resolved_config.items():
        # Count how many we already have for this class in existing_ids
        # Since existing_ids contains all downloaded ids, let's load current counts
        current_class_count = 0
        if os.path.exists(metadata_path):
            with open(metadata_path, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("class") == class_name:
                        current_class_count += 1
                        
        needed = args.target_count - current_class_count
        if needed <= 0:
            print(f"[*] Class {class_name} already has {current_class_count} images downloaded (target: {args.target_count}). Skipping query.")
            continue
            
        print(f"[*] Class {class_name} currently has {current_class_count} images. Seeking {needed} new images.")
        candidates = query_gbif_occurrences(class_name, config, args.target_count - current_class_count, existing_ids)
        all_candidates.extend(candidates)
        
    if not all_candidates:
        print("\n[*] All classes already satisfied! No new images need to be downloaded.")
        # Regenerate report in case images exist but report is missing
        generate_interactive_report(metadata_path, args.report)
        return
        
    print(f"\n[*] Discovered {len(all_candidates)} total candidate occurrences across all classes.")
    
    if args.dry_run:
        print("[*] Dry run enabled. Terminating before downloading images.")
        return
        
    # Stage 2: Parallel Download
    downloaded_records = download_images_parallel(all_candidates, args.output_dir, args.workers)
    
    # Stage 3: Save metadata and generate visual dashboard
    if downloaded_records:
        write_metadata_csv(metadata_path, downloaded_records)
        print(f"\n[*] Successfully downloaded {len(downloaded_records)} new images!")
    else:
        print("\n[!] No new images downloaded successfully.")
        
    # Generate/Regenerate the HTML dashboard report
    generate_interactive_report(metadata_path, args.report)
    
    print("\n==================================================")
    print("Pipeline Execution Completed!")
    print("==================================================")

if __name__ == "__main__":
    main()
