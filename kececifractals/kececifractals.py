# kececifractals.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import random
import math
import sys
import os # Import os to potentially handle file paths

# --- Helper Functions ---

def random_soft_color():
    """Generates a random soft RGB color tuple."""
    return tuple(random.uniform(0.4, 0.95) for _ in range(3))

def draw_circle(ax, center, radius, color):
    """Adds a circle patch to the Matplotlib axes."""
    # Using fill=True and linewidth=0 is efficient for solid circles
    ax.add_patch(Circle(center, radius, color=color, fill=True, linewidth=0))

# --- Recursive Fractal Drawing (Internal Helper) ---

def _draw_recursive_circles(ax, x, y, radius, level, max_level, num_children, min_radius, scale_factor):
    """
    Internal helper recursive function to draw child circles. Not intended for direct use.
    """
    # Base Case: Stop recursion if max level is reached
    if level > max_level:
        return

    child_radius = radius * scale_factor

    # Optimization: Stop if child circles will be too small
    if child_radius < min_radius:
        return

    # Calculate the distance from the parent center to the child centers
    distance_from_parent_center = radius - child_radius

    # --- Place and Draw Child Circles ---
    for i in range(num_children):
        # Calculate the angle for this child
        angle_rad = np.deg2rad(360 / num_children * i)

        # Calculate the center of the child circle
        child_x = x + distance_from_parent_center * np.cos(angle_rad)
        child_y = y + distance_from_parent_center * np.sin(angle_rad)

        # Draw the child circle
        child_color = random_soft_color()
        draw_circle(ax, (child_x, child_y), child_radius, child_color)

        # Recursive Call for the next level
        try:
            _draw_recursive_circles(ax, child_x, child_y, child_radius, level + 1,
                                    max_level, num_children, min_radius, scale_factor)
        except RecursionError:
            # Print a warning if recursion depth is exceeded, but continue if possible
            print(f"Warning: Maximum recursion depth likely reached near level {level+1}. "
                  f"Fractal generation may be incomplete. "
                  f"Consider reducing max_level or increasing min_size_factor.", file=sys.stderr)
            # Stop this branch of recursion if depth limit is hit
            return

# --- Main Public Function ---

def kececifractals_circle(
    initial_children=6,
    recursive_children=6,
    text="Keçeci Fractals",
    font_size=14,
    font_color='black',
    font_style='bold',
    font_family='Arial',
    max_level=4,
    min_size_factor=0.001,      # Practical minimum > 0, e.g., 0.001 or lower
    scale_factor=0.5,           # Adjusted default for potentially better visual separation
    base_radius=4.0,
    background_color=None,
    initial_circle_color=None,
    output_mode='show',         # 'show', 'png', 'svg', 'jpg'
    filename="kececi_fractal_circle-1", # Base filename for saving
    dpi=300                     # Resolution for raster formats (png, jpg)
    ):
    """
    Generates and displays or saves a Keçeci-style circle fractal.

    Args:
        initial_children (int): Number of circles in the first level.
        recursive_children (int): Number of children per circle in recursion.
        text (str): Text to display circularly around the fractal. Empty string for no text.
        font_size (int): Font size for the text.
        font_color (str): Color of the text (matplotlib color format).
        font_style (str): Font style for the text ('normal', 'bold', etc.).
        font_family (str): Font family for the text.
        max_level (int): Maximum recursion depth. Level 0 is the main circle.
        min_size_factor (float): Stop recursion if radius < base_radius * min_size_factor.
                         Must be > 0. Practical limit depends on visual needs/resources.
        scale_factor (float): Ratio of child radius to parent radius (0 < scale_factor < 1).
        base_radius (float): Radius of the initial, largest circle.
        background_color (tuple|str|None): Background color. Uses random_soft_color() if None.
        initial_circle_color (tuple|str|None): Color of the main circle. Uses random_soft_color() if None.
        output_mode (str): 'show' to display inline (Jupyter) or in a window,
                           'png', 'svg', 'jpg' to save in that format.
        filename (str): Base filename for saving (extension is added automatically).
                        The file is saved in the current working directory.
        dpi (int): Dots Per Inch resolution for saving PNG and JPG files.

    Returns:
        None: Displays the plot or saves it to a file.
    """
    # Input validation
    if not isinstance(max_level, int) or max_level < 0:
        print("Error: max_level must be a non-negative integer.", file=sys.stderr)
        return
    if not isinstance(min_size_factor, (int, float)) or min_size_factor <= 0:
        print("Error: min_size_factor must be a positive number.", file=sys.stderr)
        return
    if not isinstance(scale_factor, (int, float)) or not (0 < scale_factor < 1):
         print("Error: scale_factor must be a number between 0 and 1 (exclusive).", file=sys.stderr)
         return

    # Setup plot
    fig_size = 10 # Maintain a consistent figure size
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    # Colors
    bg_color = background_color if background_color else random_soft_color()
    fig.patch.set_facecolor(bg_color)
    main_color = initial_circle_color if initial_circle_color else random_soft_color()

    # Draw main circle (Level 0)
    draw_circle(ax, (0, 0), base_radius, main_color)

    # Calculate minimum absolute size
    min_absolute_radius = base_radius * min_size_factor

    # --- Text Placement ---
    limit = base_radius + 1.0 # Default plot limit
    if text and isinstance(text, str) and len(text) > 0:
        outer_bound_radius = base_radius
        text_radius = outer_bound_radius + 0.8 # Place text slightly outside
        for i, char in enumerate(text):
            angle_deg = (360 / len(text) * i) - 90 # Start from top
            angle_rad = np.deg2rad(angle_deg)
            x_text = text_radius * np.cos(angle_rad)
            y_text = text_radius * np.sin(angle_rad)
            rotation = angle_deg + 90 # Rotate character appropriately
            ax.text(x_text, y_text, char, fontsize=font_size, ha='center', va='center',
                    color=font_color, fontweight=font_style, fontname=font_family, rotation=rotation)
        # Adjust plot limits to ensure text is visible
        limit = max(limit, text_radius + font_size * 0.1) # Add buffer based on font size

    # --- Draw First Level (Level 1) Children and Start Recursion ---
    if max_level >= 1:
        initial_radius = base_radius * scale_factor
        # Check if the *first* level children are large enough
        if initial_radius >= min_absolute_radius:
            # Distance from the center of the main circle to the centers of the first-level children
            dist_initial = base_radius - initial_radius # = base_radius * (1 - scale_factor)

            for i in range(initial_children):
                angle_rad = np.deg2rad(360 / initial_children * i)
                # Calculate center of this initial child
                ix = 0 + dist_initial * np.cos(angle_rad)
                iy = 0 + dist_initial * np.sin(angle_rad)
                i_color = random_soft_color()

                # Draw the initial child (Level 1)
                draw_circle(ax, (ix, iy), initial_radius, i_color)

                # Start recursion for this child, beginning at Level 2
                _draw_recursive_circles(ax, ix, iy, initial_radius, level=2,
                                        max_level=max_level,
                                        num_children=recursive_children,
                                        min_radius=min_absolute_radius,
                                        scale_factor=scale_factor)

    # --- Final Plot Adjustments ---
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect('equal', adjustable='box') # Ensure circles look like circles
    ax.axis('off') # Hide the axes
    # Set title only if text is not empty, otherwise use a generic title
    plot_title = f"Keçeci Fractals ({text})" if text else "Keçeci Circle Fractal"
    plt.title(plot_title, fontsize=16)

    # --- Output Handling ---
    output_mode = output_mode.lower().strip() # Normalize output mode string

    if output_mode == 'show':
        plt.show() # Displays the plot (inline in Jupyter, or new window)
        # NOTE: No plt.close() here, as show() handles the figure lifecycle.
    elif output_mode in ['png', 'jpg', 'svg']:
        # Construct full filename in the current working directory
        output_filename = f"{filename}.{output_mode}"
        try:
            save_kwargs = {
                'bbox_inches': 'tight', # Crop whitespace
                'pad_inches': 0.1,      # Small padding
                'facecolor': fig.get_facecolor() # Ensure background color is saved
            }
            if output_mode in ['png', 'jpg']:
                 save_kwargs['dpi'] = dpi # Set resolution for raster images

            plt.savefig(output_filename, format=output_mode, **save_kwargs)
            print(f"Fractal successfully saved as '{os.path.abspath(output_filename)}'") # Show full path
        except Exception as e:
            print(f"Error saving {output_mode.upper()} file '{output_filename}': {e}", file=sys.stderr)
        finally:
             # IMPORTANT: Close the figure *after* saving or error to free memory
             plt.close(fig)
    else:
        print(f"Error: Invalid output_mode '{output_mode}'. Choose 'show', 'png', 'jpg', or 'svg'.", file=sys.stderr)
        # Close the figure if the mode was invalid and plotting happened
        plt.close(fig)

# --- Optional: Code to run only when the script is executed directly ---
# This block is useful for testing the module from the command line
# It will NOT run when the module is imported
if __name__ == "__main__":
    print(f"--- Running Test Cases for {os.path.basename(__file__)} ---")

    print("\n[Test 1: Displaying fractal inline/window (output_mode='show')]")
    kececifractals_circle(
        initial_children=5,
        recursive_children=4,
        text="Test Show",
        max_level=3,
        scale_factor=0.5,
        min_size_factor=0.001,
        output_mode='show'
    )
    print("--- Test 1 Complete ---")


    print("\n[Test 2: Saving fractal as PNG]")
    kececifractals_circle(
        initial_children=7,
        recursive_children=3,
        text="Test PNG",
        max_level=4,
        scale_factor=0.5,
        min_size_factor=0.001,
        base_radius=5,
        background_color='#101030', # Dark blue hex
        initial_circle_color='yellow',
        output_mode='png',
        filename="test_fractal_output_png-1",
        dpi=150 # Lower DPI for quick test save
    )
    print("--- Test 2 Complete ---")

    print("\n[Test 3: Saving fractal as SVG]")
    kececifractals_circle(
        initial_children=6,
        recursive_children=5,
        text="Test SVG",
        max_level=2, # Shallow level for quick test
        scale_factor=0.5,
        min_size_factor=0.001,
        output_mode='svg',
        filename="test_fractal_output_svg-1"
    )
    print("--- Test 3 Complete ---")

    print("\n[Test 4: Invalid Mode]")
    kececifractals_circle(output_mode='gif') # Intentionally invalid
    print("--- Test 4 Complete ---")

    print("\n--- All Direct Execution Tests Finished ---")
