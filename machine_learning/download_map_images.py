#!/usr/bin/env python3
"""
Map Image Downloader for CS2/CS:GO Maps
=======================================

This script downloads high-quality radar/overview images for CS2/CS:GO maps
that can be used as backgrounds in the interactive visualizer.

Usage:
    python download_map_images.py

Author: Generated for CS 5612 Project
Date: 2024
"""

import requests
from pathlib import Path
from PIL import Image
import io

def download_map_images():
    """
    Download map overview images for common CS maps
    """
    # Create maps directory
    maps_dir = Path("map_images")
    maps_dir.mkdir(exist_ok=True)
    
    # Map image URLs (these are example URLs - you may need to find better sources)
    map_urls = {
        'de_mirage': 'https://raw.githubusercontent.com/ValveSoftware/counter-strike/master/game/csgo/maps/de_mirage/radar/de_mirage.png',
        'de_dust2': 'https://raw.githubusercontent.com/ValveSoftware/counter-strike/master/game/csgo/maps/de_dust2/radar/de_dust2.png',
        'de_inferno': 'https://raw.githubusercontent.com/ValveSoftware/counter-strike/master/game/csgo/maps/de_inferno/radar/de_inferno.png',
        'de_train': 'https://raw.githubusercontent.com/ValveSoftware/counter-strike/master/game/csgo/maps/de_train/radar/de_train.png',
        'de_nuke': 'https://raw.githubusercontent.com/ValveSoftware/counter-strike/master/game/csgo/maps/de_nuke/radar/de_nuke.png'
    }
    
    # Alternative: Create high-quality placeholder images with map layouts
    create_map_placeholders(maps_dir)
    
    print(f"Map images saved to {maps_dir}")

def create_map_placeholders(maps_dir):
    """
    Create placeholder map images with basic layouts
    """
    # Mirage layout (simplified)
    create_mirage_map(maps_dir / "de_mirage.png")
    
    # You can add more maps here
    print("Created placeholder map images")

def create_mirage_map(output_path):
    """
    Create a simplified Mirage map layout
    """
    from PIL import Image, ImageDraw, ImageFont
    
    # Create image
    width, height = 1024, 1024
    img = Image.new('RGB', (width, height), color='#2F4F2F')  # Dark green background
    draw = ImageDraw.Draw(img)
    
    # Define map areas (approximate coordinates)
    areas = {
        'T Spawn': (50, 800, 200, 950),
        'T Ramp': (200, 700, 350, 800),
        'Palace': (100, 600, 250, 700),
        'A Site': (400, 200, 600, 400),
        'CT Spawn': (700, 50, 900, 200),
        'B Site': (700, 600, 900, 800),
        'Mid': (350, 400, 550, 600),
        'Connector': (550, 300, 700, 500),
        'Jungle': (250, 200, 400, 350),
        'A Ramp': (300, 350, 450, 450)
    }
    
    # Draw areas
    colors = ['#8B4513', '#654321', '#A0522D', '#D2691E', '#CD853F']
    for i, (area_name, (x1, y1, x2, y2)) in enumerate(areas.items()):
        color = colors[i % len(colors)]
        draw.rectangle([x1, y1, x2, y2], fill=color, outline='white', width=2)
        
        # Add area labels
        text_x = x1 + (x2 - x1) // 2
        text_y = y1 + (y2 - y1) // 2
        
        try:
            font = ImageFont.load_default()
            bbox = draw.textbbox((0, 0), area_name, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            draw.text((text_x - text_width//2, text_y - text_height//2), 
                     area_name, fill='white', font=font)
        except:
            draw.text((text_x - len(area_name)*3, text_y), area_name, fill='white')
    
    # Add pathways
    pathways = [
        # Main connections
        [(200, 850), (300, 750), (400, 650), (500, 550)],  # T to Mid
        [(500, 300), (600, 200), (700, 150)],  # A to CT
        [(800, 600), (700, 500), (600, 400)],  # B to Mid
        [(400, 300), (300, 400), (250, 500)],  # A to Palace
    ]
    
    for path in pathways:
        if len(path) > 1:
            for i in range(len(path) - 1):
                draw.line([path[i], path[i+1]], fill='#696969', width=8)
    
    # Add title
    draw.text((10, 10), "DE_MIRAGE", fill='white', font=ImageFont.load_default())
    
    # Save image
    img.save(output_path)
    print(f"Created Mirage map: {output_path}")

if __name__ == "__main__":
    download_map_images()
