#!/usr/bin/env python3
"""
Video Overlay Script
Takes all videos from ./runs/videos directory, overlays them with 60% opacity, and creates a combined video.

Usage Examples:
    # Basic usage with default settings (longest video as base with highest opacity)
    python get_visuals.py
    
    # Make all videos clearly visible (recommended for better visibility)
    python get_visuals.py --blend-mode visible --opacity 0.6
    
    # Specify custom input directory and output file
    python get_visuals.py --input-dir ./dqn_pytorch/videos --output my_overlay.mp4
    
    # Limit to first 20 videos with 50% opacity using alpha blending
    python get_visuals.py --max-videos 20 --opacity 0.5 --blend-mode alpha
    
    # Process all videos with weighted blending
    python get_visuals.py -i ./my_videos -o combined.mp4 -p 0.6 -m 50 -b weighted
    
    # Use average blending to prevent over-brightening
    python get_visuals.py --blend-mode average --opacity 0.6
"""

import cv2
import numpy as np
import os
import glob
from pathlib import Path
import argparse


def get_video_files(video_dir):
    """Get all video files from the specified directory."""
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv', '*.flv']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(video_dir, ext)))
    
    return sorted(video_files)


def get_video_properties(video_path):
    """Get video properties (width, height, fps, frame_count)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    cap.release()
    return width, height, fps, frame_count


def find_longest_video(video_files):
    """Find the video with the longest duration (most frames)."""
    longest_video = None
    max_frames = 0
    
    for video_file in video_files:
        props = get_video_properties(video_file)
        if props:
            width, height, fps, frame_count = props
            if frame_count > max_frames:
                max_frames = frame_count
                longest_video = video_file
    
    return longest_video, max_frames


def overlay_videos(video_files, output_path, opacity=0.6, max_videos=None, blend_mode='longest_base'):
    """
    Overlay multiple videos with specified opacity.
    
    Args:
        video_files: List of video file paths
        output_path: Output video file path
        opacity: Opacity for overlay (0.0 to 1.0)
        max_videos: Maximum number of videos to process (None for all)
        blend_mode: Blending mode ('longest_base', 'average', 'alpha', 'weighted')
    """
    if not video_files:
        print("No video files found!")
        return False
    
    # Limit number of videos if specified
    if max_videos and len(video_files) > max_videos:
        print(f"Limiting to first {max_videos} videos out of {len(video_files)} found")
        video_files = video_files[:max_videos]
    
    print(f"Processing {len(video_files)} video files")
    
    # Find the longest video to use as base
    if blend_mode == 'longest_base':
        longest_video, max_frames = find_longest_video(video_files)
        if not longest_video:
            print("Could not find any valid video files!")
            return False
        
        # Move longest video to front of the list
        video_files = [longest_video] + [v for v in video_files if v != longest_video]
        print(f"Using longest video as base: {os.path.basename(longest_video)} ({max_frames} frames)")
    
    # Get properties from the first video (now the longest one)
    first_video_props = get_video_properties(video_files[0])
    if not first_video_props:
        print(f"Could not read video: {video_files[0]}")
        return False
    
    width, height, fps, frame_count = first_video_props
    print(f"Video properties: {width}x{height}, {fps} FPS, {frame_count} frames")
    
    # Initialize video writers for each video
    video_caps = []
    for i, video_file in enumerate(video_files):
        cap = cv2.VideoCapture(video_file)
        if cap.isOpened():
            video_caps.append(cap)
            print(f"Loaded video {i+1}/{len(video_files)}: {os.path.basename(video_file)}")
        else:
            print(f"Warning: Could not open video: {video_file}")
    
    if not video_caps:
        print("No valid video files could be opened!")
        return False
    
    # Set up output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print("Could not create output video writer!")
        for cap in video_caps:
            cap.release()
        return False
    
    # Process frames
    frame_idx = 0
    max_frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in video_caps)
    
    print(f"Processing {max_frames} frames...")
    
    while frame_idx < max_frames:
        frames = []
        
        # Read frame from each video
        for i, cap in enumerate(video_caps):
            ret, frame = cap.read()
            if ret:
                # Resize frame to match first video dimensions
                frame = cv2.resize(frame, (width, height))
                frames.append(frame)
            else:
                print(f"Warning: Could not read frame {frame_idx} from video {i}")
        
        if not frames:
            break
        
        # Apply different blending modes
        if blend_mode == 'longest_base':
            # Use longest video (first frame) as base with higher weight, others with good visibility
            float_frames = [frame.astype(np.float32) for frame in frames]
            
            # Give the longest video (first) more weight, but make others clearly visible
            weights = [1.0] + [opacity * 2.0] * (len(frames) - 1)  # Double opacity for better visibility
            total_weight = sum(weights)
            
            combined_frame = np.zeros_like(float_frames[0])
            for frame, weight in zip(float_frames, weights):
                combined_frame += frame * weight
            combined_frame = combined_frame / total_weight
            
            combined_frame = np.clip(combined_frame, 0, 255).astype(np.uint8)
            
        elif blend_mode == 'average':
            # Simple average of all frames - prevents over-brightening
            # float_frames = [frame.astype(np.float32) for frame in frames]
            # combined_frame = np.mean(float_frames, axis=0)
            # combined_frame = np.clip(combined_frame, 0, 255).astype(np.uint8)

            float_frames = [f.astype(np.float32) for f in frames]
            combined = np.maximum.reduce(float_frames)
            combined_frame = combined.clip(0, 255).astype(np.uint8)
            
        elif blend_mode == 'subtracted':
            
            f32 = [f.astype(np.float32) for f in frames]
            bg = np.median(f32, axis=0)                    # robust background
            trails = [np.abs(f - bg) for f in f32]
            trail_sum = np.sum(trails, axis=0)
            p99 = np.percentile(trail_sum, 99)
            trail_vis = np.clip(trail_sum / max(p99,1e-6), 0, 1)  # 0..1

            # Blend trails back onto background for context
            alpha = 0.85
            combined = (1 - alpha) * (bg/255.0) + alpha * trail_vis
            combined_frame = (np.clip(combined, 0, 1) * 255).astype(np.uint8)

        elif blend_mode == 'alpha':
            # Alpha blending with distributed opacity
            combined_frame = frames[0].copy().astype(np.float32)
            for i, frame in enumerate(frames[1:], 1):
                frame_float = frame.astype(np.float32)
                alpha = opacity / len(frames)
                combined_frame = alpha * frame_float + (1 - alpha) * combined_frame
            combined_frame = np.clip(combined_frame, 0, 255).astype(np.uint8)
            
        elif blend_mode == 'weighted':
            # Weighted average with first frame having more weight
            float_frames = [frame.astype(np.float32) for frame in frames]
            weights = [1.0] + [opacity] * (len(frames) - 1)
            total_weight = sum(weights)
            
            combined_frame = np.zeros_like(float_frames[0])
            for frame, weight in zip(float_frames, weights):
                combined_frame += frame * weight
            combined_frame = combined_frame / total_weight
            combined_frame = np.clip(combined_frame, 0, 255).astype(np.uint8)
            
        elif blend_mode == 'visible':
            # Make all videos clearly visible using screen blending
            float_frames = [frame.astype(np.float32) for frame in frames]
            
            # Start with the longest video (first frame)
            combined_frame = float_frames[0].copy()
            
            # Use screen blending for better visibility of all videos
            for frame in float_frames[1:]:
                # Screen blending: 1 - (1 - a) * (1 - b)
                combined_frame = 1.0 - (1.0 - combined_frame/255.0) * (1.0 - frame*opacity/255.0)
                combined_frame = combined_frame * 255.0
            
            combined_frame = np.clip(combined_frame, 0, 255).astype(np.uint8)
            
        elif blend_mode == 'darker':
            # Darker blending mode - uses multiply blending for richer, darker results
            float_frames = [frame.astype(np.float32) for frame in frames]
            
            # Start with the longest video (first frame)
            combined_frame = float_frames[0].copy()
            
            # Use multiply blending for darker, more saturated results
            for i, frame in enumerate(float_frames[1:], 1):
                # Multiply blending: a * b (normalized to 0-1 range)
                # Apply opacity to the overlay frame
                overlay_frame = frame * opacity
                
                # Multiply blending: darker results, preserves details
                combined_frame = (combined_frame / 255.0) * (overlay_frame / 255.0) * 255.0
            
            # Apply a slight gamma correction to brighten the result while keeping it darker
            combined_frame = np.power(combined_frame / 255.0, 0.8) * 255.0
            combined_frame = np.clip(combined_frame, 0, 255).astype(np.uint8)
        
        # Write the combined frame
        out.write(combined_frame)
        
        frame_idx += 1
        
        # Progress indicator
        if frame_idx % 50 == 0 or frame_idx == max_frames:
            print(f"Processed {frame_idx}/{max_frames} frames ({frame_idx/max_frames*100:.1f}%)")
    
    # Clean up
    for cap in video_caps:
        cap.release()
    out.release()
    
    print(f"Video overlay complete! Output saved to: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description='Overlay multiple videos with specified opacity')
    parser.add_argument('--input-dir', '-i', default='./runs/videos', 
                       help='Input directory containing videos (default: ./runs/videos)')
    parser.add_argument('--output', '-o', default='overlayed_video.mp4',
                       help='Output video file name (default: overlayed_video.mp4)')
    parser.add_argument('--opacity', '-p', type=float, default=0.6,
                       help='Opacity for overlay (0.0 to 1.0, default: 0.6)')
    parser.add_argument('--max-videos', '-m', type=int, default=None,
                       help='Maximum number of videos to process (default: all)')
    parser.add_argument('--blend-mode', '-b', choices=['longest_base', 'visible', 'average', 'alpha', 'weighted', 'darker', 'subtracted'], 
                       default='longest_base', help='Blending mode (default: longest_base)')
    
    args = parser.parse_args()
    
    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist!")
        return 1
    
    # Get video files
    video_files = get_video_files(args.input_dir)
    
    if not video_files:
        print(f"No video files found in '{args.input_dir}'")
        return 1
    
    print(f"Input directory: {args.input_dir}")
    print(f"Output file: {args.output}")
    print(f"Opacity: {args.opacity}")
    print(f"Blend mode: {args.blend_mode}")
    if args.max_videos:
        print(f"Max videos: {args.max_videos}")
    
    # Create output video
    success = overlay_videos(video_files, args.output, args.opacity, args.max_videos, args.blend_mode)
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
