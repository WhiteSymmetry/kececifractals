# kececifractals.py
"""
This module provides two primary functionalities for generating Keçeci Fractals:
1.  kececifractals_circle(): Generates general-purpose, aesthetic, and randomly
    colored circular fractals.
2.  visualize_qec_fractal(): Generates fractals customized for modeling the
    concept of Quantum Error Correction (QEC) codes.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import random
import math
import sys
import os

# --- GENERAL HELPER FUNCTIONS ---

def random_soft_color():
    """Generates a random soft RGB color tuple."""
    return tuple(random.uniform(0.4, 0.95) for _ in range(3))

def _draw_circle_patch(ax, center, radius, face_color, edge_color='black', lw=0.5):
    """
    A robust helper function that adds a circle patch to the Matplotlib axes,
    using facecolor and edgecolor to avoid the UserWarning.
    """
    ax.add_patch(Circle(center, radius, facecolor=face_color, edgecolor=edge_color, linewidth=lw, fill=True))


# ==============================================================================
# PART 1: GENERAL-PURPOSE KEÇECİ FRACTALS
# ==============================================================================

def _draw_recursive_circles(ax, x, y, radius, level, max_level, num_children, min_radius, scale_factor):
    """
    Internal recursive helper function to draw child circles for general fractals.
    Not intended for direct use.
    """
    if level > max_level:
        return

    child_radius = radius * scale_factor
    if child_radius < min_radius:
        return

    distance_from_parent_center = radius - child_radius

    for i in range(num_children):
        angle_rad = np.deg2rad(360 / num_children * i)
        child_x = x + distance_from_parent_center * np.cos(angle_rad)
        child_y = y + distance_from_parent_center * np.sin(angle_rad)
        
        child_color = random_soft_color()
        # General-purpose fractal uses lw=0 for solid, borderless circles.
        _draw_circle_patch(ax, (child_x, child_y), child_radius, face_color=child_color, lw=0)

        try:
            _draw_recursive_circles(ax, child_x, child_y, child_radius, level + 1,
                                    max_level, num_children, min_radius, scale_factor)
        except RecursionError:
            print(f"Warning: Maximum recursion depth reached. Fractal may be incomplete.", file=sys.stderr)
            return

def kececifractals_circle(
    initial_children=6, recursive_children=6, text="Keçeci Fractals",
    font_size=14, font_color='black', font_style='bold', font_family='Arial',
    max_level=4, min_size_factor=0.001, scale_factor=0.5,
    base_radius=4.0, background_color=None, initial_circle_color=None,
    output_mode='show', filename="kececi_fractal_circle", dpi=300
    ):
    """
    Generates, displays, or saves a general-purpose, aesthetic Keçeci-style circle fractal.
    """
    # Input validation
    if not isinstance(max_level, int) or max_level < 0:
        print("Error: max_level must be a non-negative integer.", file=sys.stderr)
        return
    if not (0 < scale_factor < 1):
         print("Error: scale_factor must be a number between 0 and 1 (exclusive).", file=sys.stderr)
         return

    fig, ax = plt.subplots(figsize=(10, 10))
    bg_color = background_color if background_color else random_soft_color()
    fig.patch.set_facecolor(bg_color)
    main_color = initial_circle_color if initial_circle_color else random_soft_color()

    # Draw the main circle
    _draw_circle_patch(ax, (0, 0), base_radius, face_color=main_color, lw=0)

    min_absolute_radius = base_radius * min_size_factor
    limit = base_radius + 1.0

    # Text placement
    if text and isinstance(text, str) and len(text) > 0:
        text_radius = base_radius + 0.8
        for i, char in enumerate(text):
            angle_deg = (360 / len(text) * i) - 90
            angle_rad = np.deg2rad(angle_deg)
            x_text, y_text = text_radius * np.cos(angle_rad), text_radius * np.sin(angle_rad)
            ax.text(x_text, y_text, char, fontsize=font_size, ha='center', va='center',
                    color=font_color, fontweight=font_style, fontname=font_family, rotation=angle_deg + 90)
        limit = max(limit, text_radius + font_size * 0.1)

    # Start the recursion
    if max_level >= 1:
        initial_radius = base_radius * scale_factor
        if initial_radius >= min_absolute_radius:
            dist_initial = base_radius - initial_radius
            for i in range(initial_children):
                angle_rad = np.deg2rad(360 / initial_children * i)
                ix, iy = dist_initial * np.cos(angle_rad), dist_initial * np.sin(angle_rad)
                i_color = random_soft_color()
                _draw_circle_patch(ax, (ix, iy), initial_radius, face_color=i_color, lw=0)
                _draw_recursive_circles(ax, ix, iy, initial_radius, 2, max_level,
                                        recursive_children, min_absolute_radius, scale_factor)

    # Plot adjustments
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    plot_title = f"Keçeci Fractals ({text})" if text else "Keçeci Circle Fractal"
    plt.title(plot_title, fontsize=16)

    # Output handling
    output_mode = output_mode.lower().strip()
    if output_mode == 'show':
        plt.show()
    elif output_mode in ['png', 'jpg', 'svg']:
        output_filename = f"{filename}.{output_mode}"
        try:
            save_kwargs = {'bbox_inches': 'tight', 'pad_inches': 0.1, 'facecolor': fig.get_facecolor()}
            if output_mode in ['png', 'jpg']:
                 save_kwargs['dpi'] = dpi
            plt.savefig(output_filename, format=output_mode, **save_kwargs)
            print(f"Fractal successfully saved to: '{os.path.abspath(output_filename)}'")
        except Exception as e:
            print(f"Error: Could not save file '{output_filename}': {e}", file=sys.stderr)
        finally:
             plt.close(fig)
    else:
        print(f"Error: Invalid output_mode '{output_mode}'. Choose 'show', 'png', 'jpg', or 'svg'.", file=sys.stderr)
        plt.close(fig)


# ==============================================================================
# PART 2: QUANTUM ERROR CORRECTION (QEC) VISUALIZATION
# ==============================================================================

def _draw_recursive_qec(ax, x, y, radius, level, max_level, num_children, scale_factor, 
                        physical_qubit_color, error_color, error_qubits, current_path):
    """
    Internal recursive function to draw physical qubits and check for errors for the QEC model.
    """
    if level > max_level:
        return

    child_radius = radius * scale_factor
    distance_from_parent_center = radius * (1 - scale_factor)

    for i in range(num_children):
        child_path = current_path + [i]
        angle_rad = np.deg2rad(360 / num_children * i)
        child_x = x + distance_from_parent_center * np.cos(angle_rad)
        child_y = y + distance_from_parent_center * np.sin(angle_rad)
        
        qubit_color = error_color if child_path in error_qubits else physical_qubit_color
        _draw_circle_patch(ax, (child_x, child_y), child_radius, face_color=qubit_color, lw=0.75)

        _draw_recursive_qec(ax, child_x, child_y, child_radius, level + 1,
                            max_level, num_children, scale_factor,
                            physical_qubit_color, error_color, error_qubits, child_path)

def visualize_qec_fractal(
    physical_qubits_per_level: int = 5, recursion_level: int = 1,
    error_qubits: list = None,
    logical_qubit_color: str = '#4A90E2',      # Blue
    physical_qubit_color: str = '#E0E0E0',    # Light Gray
    error_color: str = '#D0021B',             # Red
    background_color: str = '#1C1C1C',        # Dark Gray
    scale_factor: float = 0.5, filename: str = "qec_fractal_visualization",
    dpi: int = 300
    ):
    """
    Visualizes a Quantum Error Correction (QEC) code concept using Keçeci Fractals.
    """
    error_qubits = [] if error_qubits is None else error_qubits

    fig, ax = plt.subplots(figsize=(12, 12))
    fig.patch.set_facecolor(background_color)
    
    base_radius = 5.0

    # Draw the Logical Qubit
    _draw_circle_patch(ax, (0, 0), base_radius, face_color=logical_qubit_color, lw=1.5)
    ax.text(0, 0, 'L', color='white', ha='center', va='center', fontsize=40, fontweight='bold', fontfamily='sans-serif')

    # Draw the Physical Qubits
    if recursion_level >= 1:
        initial_radius = base_radius * scale_factor
        dist_initial = base_radius * (1 - scale_factor)
        for i in range(physical_qubits_per_level):
            child_path = [i]
            angle_rad = np.deg2rad(360 / physical_qubits_per_level * i)
            ix, iy = dist_initial * np.cos(angle_rad), dist_initial * np.sin(angle_rad)
            qubit_color = error_color if child_path in error_qubits else physical_qubit_color
            
            _draw_circle_patch(ax, (ix, iy), initial_radius, face_color=qubit_color, lw=0.75)
            # Add a number label to the first-level qubits for clarity
            ax.text(ix, iy, str(i), color='black', ha='center', va='center', fontsize=12, fontweight='bold')

            _draw_recursive_qec(ax, ix, iy, initial_radius, 2, recursion_level,
                                physical_qubits_per_level, scale_factor,
                                physical_qubit_color, error_color, error_qubits, child_path)

    # Finalize and Save the Plot
    ax.set_xlim(-base_radius - 1.5, base_radius + 1.5)
    ax.set_ylim(-base_radius - 1.5, base_radius + 1.5)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    
    title = f"QEC Fractal Model: {physical_qubits_per_level}-Qubit Code | Level: {recursion_level} | Errors: {len(error_qubits)}"
    plt.title(title, color='white', fontsize=18, pad=20)
    
    output_filename = f"{filename}.png"
    plt.savefig(output_filename, format='png', dpi=dpi, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Visualization saved to: '{os.path.abspath(output_filename)}'")


# ==============================================================================
# PART 3: MODULE TESTS
# ==============================================================================

if __name__ == "__main__":
    print(f"--- Running Test Cases for {os.path.basename(__file__)} ---")

    # --- General-Purpose Fractal Tests ---
    print("\n--- PART 1: General-Purpose Fractal Tests ---")
    print("\n[Test 1.1: Displaying fractal on screen (show)]")
    kececifractals_circle(
        initial_children=5,
        recursive_children=4,
        text="Test Show",
        max_level=3,
        output_mode='show'
    )
    
    print("\n[Test 1.2: Saving fractal as PNG]")
    kececifractals_circle(
        initial_children=7,
        recursive_children=3,
        text="Test PNG Save",
        background_color='#101030',
        initial_circle_color='yellow',
        output_mode='png',
        filename="test_fractal_generic"
    )

    # --- QEC Visualization Tests ---
    print("\n--- PART 2: QEC Visualization Tests ---")
    print("\n[Test 2.1: Generating an error-free 7-qubit code...]")
    visualize_qec_fractal(
        physical_qubits_per_level=7,
        recursion_level=1,
        error_qubits=[],
        filename="QEC_Model_Test_No_Errors"
    )

    print("\n[Test 2.2: Generating a 7-qubit code with a single error...]")
    visualize_qec_fractal(
        physical_qubits_per_level=7,
        recursion_level=1,
        error_qubits=[[3]],
        filename="QEC_Model_Test_Single_Error"
    )
    
    print("\n[Test 2.3: Generating a 2-level code with a deep-level error...]")
    visualize_qec_fractal(
        physical_qubits_per_level=5,
        recursion_level=2,
        error_qubits=[[4, 1]],
        filename="QEC_Model_Test_Deep_Error"
    )

    print("\n--- All Tests Completed ---")
