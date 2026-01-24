# kececifractals.py
"""
This module provides three primary functionalities for generating Keçeci Fractals:
1.  kececifractals_circle(): Generates general-purpose, aesthetic, and randomly
    colored circular fractals.
2.  visualize_qec_fractal(): Generates fractals customized for modeling the
    concept of Quantum Error Correction (QEC) codes.
3.  kececifractals_3d(): Generates 3D versions of Keçeci fractals.
"""

import math
import os
import random
import sys
from typing import Callable, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import networkx as nx  # STRATUM MODEL VISUALIZATION
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D, art3d

# Import kececilayout if available, otherwise use a fallback
try:
    import kececilayout as kl  # STRATUM MODEL VISUALIZATION
except ImportError:
    # Fallback layout function if kececilayout is not available
    class kl:
        @staticmethod
        def kececi_layout(
            G, primary_direction="top_down", primary_spacing=1.5, secondary_spacing=1.0
        ):
            pos = {}
            for i, node in enumerate(G.nodes()):
                if primary_direction == "top_down":
                    pos[node] = (i * secondary_spacing, -i * primary_spacing)
                else:
                    pos[node] = (i * primary_spacing, i * secondary_spacing)
            return pos


# --- GENERAL HELPER FUNCTIONS ---


def random_soft_color():
    """Generates a random soft RGB color tuple."""
    return tuple(random.uniform(0.4, 0.95) for _ in range(3))


def _parse_color(
    color_input: Union[str, Tuple[float, float, float], None],
) -> Optional[Tuple[float, float, float]]:
    """
    Parses color input which can be:
    - None
    - RGB tuple (0-1 range)
    - Hex string like '#RRGGBB'
    - Named color like 'red', 'blue', etc.

    Returns RGB tuple in 0-1 range or None.
    """
    if color_input is None:
        return None

    # If already a tuple, assume it's correct format
    if isinstance(color_input, tuple):
        if len(color_input) == 3:
            return color_input
        elif len(color_input) == 4:
            return color_input[:3]  # Drop alpha if present

    # Try to parse as string
    if isinstance(color_input, str):
        try:
            # First try matplotlib's color conversion
            rgb = to_rgb(color_input)
            return rgb
        except (ValueError, AttributeError):
            # Try hex parsing manually
            if color_input.startswith("#"):
                try:
                    # Remove # and parse
                    hex_color = color_input.lstrip("#")
                    if len(hex_color) == 3:
                        # Expand shorthand #RGB to #RRGGBB
                        hex_color = "".join([c * 2 for c in hex_color])
                    elif len(hex_color) != 6:
                        raise ValueError(f"Invalid hex color: {color_input}")

                    # Convert to RGB 0-255
                    r = int(hex_color[0:2], 16) / 255.0
                    g = int(hex_color[2:4], 16) / 255.0
                    b = int(hex_color[4:6], 16) / 255.0
                    return (r, g, b)
                except:
                    pass

    # If we get here, return random color as fallback
    print(
        f"Warning: Could not parse color '{color_input}'. Using random color.",
        file=sys.stderr,
    )
    return random_soft_color()


def _draw_circle_patch(ax, center, radius, face_color, edge_color="black", lw=0.5):
    """
    A robust helper function that adds a circle patch to the Matplotlib axes,
    using facecolor and edgecolor to avoid the UserWarning.
    """
    ax.add_patch(
        Circle(
            center,
            radius,
            facecolor=face_color,
            edgecolor=edge_color,
            linewidth=lw,
            fill=True,
        )
    )


# ==============================================================================
# PART 1: GENERAL-PURPOSE KEÇECİ FRACTALS
# ==============================================================================


def _draw_recursive_circles(
    ax, x, y, radius, level, max_level, num_children, min_radius, scale_factor
):
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
        _draw_circle_patch(
            ax, (child_x, child_y), child_radius, face_color=child_color, lw=0
        )

        try:
            _draw_recursive_circles(
                ax,
                child_x,
                child_y,
                child_radius,
                level + 1,
                max_level,
                num_children,
                min_radius,
                scale_factor,
            )
        except RecursionError:
            print(
                "Warning: Maximum recursion depth reached. Fractal may be incomplete.",
                file=sys.stderr,
            )
            return


def kececifractals_circle(
    initial_children: int = 6,
    recursive_children: int = 6,
    text: str = "Keçeci Fractals",
    font_size: int = 14,
    font_color: str = "black",
    font_style: str = "bold",
    font_family: str = "Arial",
    max_level: int = 4,
    min_size_factor: float = 0.001,
    scale_factor: float = 0.5,
    base_radius: float = 4.0,
    background_color: Union[str, Tuple[float, float, float], None] = None,
    initial_circle_color: Union[str, Tuple[float, float, float], None] = None,
    output_mode: str = "show",
    filename: str = "kececi_fractal_circle",
    dpi: int = 300,
) -> None:
    """
    Generates, displays, or saves a general-purpose, aesthetic Keçeci-style circle fractal.

    Args:
        initial_children: Number of first-level child circles
        recursive_children: Number of children for deeper levels
        text: Text to display around the fractal
        font_size: Font size for text
        font_color: Color of text (string or hex)
        font_style: Font style ('normal', 'bold', 'italic', etc.)
        font_family: Font family name
        max_level: Maximum recursion depth
        min_size_factor: Minimum radius as factor of base_radius
        scale_factor: Size reduction factor for child circles
        base_radius: Radius of the central circle
        background_color: Background color (hex string, named color, or RGB tuple)
        initial_circle_color: Color of central circle (hex string, named color, or RGB tuple)
        output_mode: 'show' or file format ('png', 'jpg', etc.)
        filename: Base filename for saving
        dpi: DPI for saved images
    """
    # Input validation
    if not isinstance(max_level, int) or max_level < 0:
        print("Error: max_level must be a non-negative integer.", file=sys.stderr)
        return
    if not (0 < scale_factor < 1):
        print(
            "Error: scale_factor must be a number between 0 and 1 (exclusive).",
            file=sys.stderr,
        )
        return

    fig, ax = plt.subplots(figsize=(10, 10))

    # Parse colors (accepts hex strings, named colors, or RGB tuples)
    bg_color = _parse_color(background_color) or random_soft_color()
    main_color = _parse_color(initial_circle_color) or random_soft_color()

    # Parse font color
    parsed_font_color = _parse_color(font_color) or (0, 0, 0)

    fig.patch.set_facecolor(bg_color)

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
            x_text, y_text = text_radius * np.cos(angle_rad), text_radius * np.sin(
                angle_rad
            )
            ax.text(
                x_text,
                y_text,
                char,
                fontsize=font_size,
                ha="center",
                va="center",
                color=parsed_font_color,
                fontweight=font_style,
                fontfamily=font_family,
                rotation=angle_deg + 90,
            )
        limit = max(limit, text_radius + font_size * 0.1)

    # Start the recursion
    if max_level >= 1:
        initial_radius = base_radius * scale_factor
        if initial_radius >= min_absolute_radius:
            dist_initial = base_radius - initial_radius
            for i in range(initial_children):
                angle_rad = np.deg2rad(360 / initial_children * i)
                ix, iy = dist_initial * np.cos(angle_rad), dist_initial * np.sin(
                    angle_rad
                )
                i_color = random_soft_color()
                _draw_circle_patch(
                    ax, (ix, iy), initial_radius, face_color=i_color, lw=0
                )
                _draw_recursive_circles(
                    ax,
                    ix,
                    iy,
                    initial_radius,
                    2,
                    max_level,
                    recursive_children,
                    min_absolute_radius,
                    scale_factor,
                )

    # Plot adjustments
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    plot_title = f"Keçeci Fractals ({text})" if text else "Keçeci Circle Fractal"
    plt.title(plot_title, fontsize=16)

    # Output handling
    output_mode = output_mode.lower().strip()
    if output_mode == "show":
        plt.show()
    elif output_mode in ["png", "jpg", "jpeg", "svg"]:
        output_filename = f"{filename}.{output_mode}"
        try:
            save_kwargs = {
                "bbox_inches": "tight",
                "pad_inches": 0.1,
                "facecolor": fig.get_facecolor(),
            }
            if output_mode in ["png", "jpg", "jpeg"]:
                save_kwargs["dpi"] = dpi
            plt.savefig(output_filename, format=output_mode, **save_kwargs)
            print(
                f"Fractal successfully saved to: '{os.path.abspath(output_filename)}'"
            )
        except Exception as e:
            print(
                f"Error: Could not save file '{output_filename}': {e}", file=sys.stderr
            )
        finally:
            plt.close(fig)
    else:
        print(
            f"Error: Invalid output_mode '{output_mode}'. Choose 'show', 'png', 'jpg', or 'svg'.",
            file=sys.stderr,
        )
        plt.close(fig)


# ==============================================================================
# PART 2: QUANTUM ERROR CORRECTION (QEC) VISUALIZATION
# ==============================================================================


def _draw_recursive_qec(
    ax,
    x,
    y,
    radius,
    level,
    max_level,
    num_children,
    scale_factor,
    physical_qubit_color,
    error_color,
    error_qubits,
    current_path,
):
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

        qubit_color = (
            error_color if child_path in error_qubits else physical_qubit_color
        )
        _draw_circle_patch(
            ax, (child_x, child_y), child_radius, face_color=qubit_color, lw=0.75
        )

        _draw_recursive_qec(
            ax,
            child_x,
            child_y,
            child_radius,
            level + 1,
            max_level,
            num_children,
            scale_factor,
            physical_qubit_color,
            error_color,
            error_qubits,
            child_path,
        )


def visualize_qec_fractal(
    physical_qubits_per_level: int = 5,
    recursion_level: int = 1,
    error_qubits: Optional[List[List[int]]] = None,
    logical_qubit_color: str = "#4A90E2",  # Blue
    physical_qubit_color: str = "#E0E0E0",  # Light Gray
    error_color: str = "#D0021B",  # Red
    background_color: str = "#1C1C1C",  # Dark Gray
    scale_factor: float = 0.5,
    filename: str = "qec_fractal_visualization",
    dpi: int = 300,
) -> None:
    """
    Visualizes a Quantum Error Correction (QEC) code concept using Keçeci Fractals.
    """
    error_qubits = [] if error_qubits is None else error_qubits

    fig, ax = plt.subplots(figsize=(12, 12))

    # Parse colors for QEC visualization
    logical_color_parsed = _parse_color(logical_qubit_color) or (
        0.29,
        0.56,
        0.89,
    )  # Default blue
    physical_color_parsed = _parse_color(physical_qubit_color) or (
        0.88,
        0.88,
        0.88,
    )  # Default light gray
    error_color_parsed = _parse_color(error_color) or (0.82, 0.01, 0.11)  # Default red
    bg_color_parsed = _parse_color(background_color) or (
        0.11,
        0.11,
        0.11,
    )  # Default dark gray

    fig.patch.set_facecolor(bg_color_parsed)

    base_radius = 5.0

    # Draw the Logical Qubit
    _draw_circle_patch(ax, (0, 0), base_radius, face_color=logical_color_parsed, lw=1.5)
    ax.text(
        0,
        0,
        "L",
        color="white",
        ha="center",
        va="center",
        fontsize=40,
        fontweight="bold",
        fontfamily="sans-serif",
    )

    # Draw the Physical Qubits
    if recursion_level >= 1:
        initial_radius = base_radius * scale_factor
        dist_initial = base_radius * (1 - scale_factor)
        for i in range(physical_qubits_per_level):
            child_path = [i]
            angle_rad = np.deg2rad(360 / physical_qubits_per_level * i)
            ix, iy = dist_initial * np.cos(angle_rad), dist_initial * np.sin(angle_rad)
            qubit_color = (
                error_color_parsed
                if child_path in error_qubits
                else physical_color_parsed
            )

            _draw_circle_patch(
                ax, (ix, iy), initial_radius, face_color=qubit_color, lw=0.75
            )
            # Add a number label to the first-level qubits for clarity
            label_color = "black" if qubit_color != error_color_parsed else "white"
            ax.text(
                ix,
                iy,
                str(i),
                color=label_color,
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
            )

            _draw_recursive_qec(
                ax,
                ix,
                iy,
                initial_radius,
                2,
                recursion_level,
                physical_qubits_per_level,
                scale_factor,
                physical_color_parsed,
                error_color_parsed,
                error_qubits,
                child_path,
            )

    # Finalize and Save the Plot
    ax.set_xlim(-base_radius - 1.5, base_radius + 1.5)
    ax.set_ylim(-base_radius - 1.5, base_radius + 1.5)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    title = f"QEC Fractal Model: {physical_qubits_per_level}-Qubit Code | Level: {recursion_level} | Errors: {len(error_qubits)}"
    plt.title(title, color="white", fontsize=18, pad=20)

    output_filename = f"{filename}.png"
    plt.savefig(
        output_filename,
        format="png",
        dpi=dpi,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)
    print(f"Visualization saved to: '{os.path.abspath(output_filename)}'")


# ==============================================================================
# PART 3: 3D KEÇECİ FRACTALS
# ==============================================================================

try:
    from mpl_toolkits.mplot3d import Axes3D, art3d

    HAS_3D = True
except ImportError:
    HAS_3D = False
    print(
        "Warning: 3D plotting not available. Install matplotlib for 3D support.",
        file=sys.stderr,
    )


def _draw_3d_sphere(ax, center, radius, color, alpha=1.0):
    """
    Draws a 3D sphere on the given axes.
    """
    if not HAS_3D:
        return

    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)

    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_surface(
        x,
        y,
        z,
        color=color,
        alpha=alpha,
        edgecolor="none",
        antialiased=True,
        shade=True,
    )


def _create_recursive_3d_fractal(
    ax,
    center,
    radius,
    level,
    max_level,
    num_children,
    scale_factor,
    min_radius,
    color_func,
    alpha_decay,
):
    """
    Recursive function to create 3D fractal spheres.
    """
    if level > max_level or radius < min_radius:
        return

    # Draw current sphere
    color = color_func(level)
    alpha = 1.0 * (alpha_decay**level)
    _draw_3d_sphere(ax, center, radius, color, alpha)

    # Calculate positions for child spheres
    child_radius = radius * scale_factor
    if child_radius < min_radius:
        return

    # For 3D, distribute children on a sphere surface
    phi = np.pi * (3.0 - np.sqrt(5.0))  # Golden angle

    for i in range(num_children):
        # Fibonacci sphere distribution for even spacing
        y = 1 - (i / float(num_children - 1)) * 2
        radius_xy = np.sqrt(1 - y * y)

        theta = phi * i

        x = np.cos(theta) * radius_xy
        z = np.sin(theta) * radius_xy

        # Scale to put children on surface of parent sphere
        direction = np.array([x, y, z])
        direction = direction / np.linalg.norm(direction)

        child_center = center + direction * (radius + child_radius)

        # Recursive call
        _create_recursive_3d_fractal(
            ax,
            child_center,
            child_radius,
            level + 1,
            max_level,
            num_children,
            scale_factor,
            min_radius,
            color_func,
            alpha_decay,
        )


def get_cmap_safe(cmap_name: str):
    """Güvenli colormap alımı, tüm matplotlib sürümleriyle uyumlu"""
    try:
        # Matplotlib 3.7+ için modern yöntem
        return plt.colormaps[cmap_name]
    except (AttributeError, KeyError):
        try:
            # Klasik yöntem
            return plt.get_cmap(cmap_name)
        except:
            # Son çare olarak plt.cm
            import matplotlib.cm as cm

            return cm.get_cmap(cmap_name)


def kececifractals_3d(
    num_children: int = 8,
    max_level: int = 3,
    scale_factor: float = 0.4,
    base_radius: float = 1.0,
    min_radius: float = 0.05,
    color_scheme: str = "plasma",
    alpha_decay: float = 0.7,
    figsize: Tuple[int, int] = (12, 10),
    elev: float = 30,
    azim: float = 45,
    output_mode: str = "show",
    filename: str = "kececi_fractal_3d",
    dpi: int = 300,
) -> None:
    """
    Generates a 3D version of Keçeci fractals.

    Args:
        num_children: Number of child spheres at each level
        max_level: Maximum recursion depth
        scale_factor: Size reduction factor for child spheres
        base_radius: Radius of the central sphere
        min_radius: Minimum sphere radius (stops recursion when reached)
        color_scheme: Matplotlib colormap name
        alpha_decay: Alpha transparency decay factor per level
        figsize: Figure size (width, height)
        elev: Elevation angle for 3D view
        azim: Azimuth angle for 3D view
        output_mode: 'show' or file format ('png', 'jpg', etc.)
        filename: Base filename for saving
        dpi: DPI for saved images
    """
    if not HAS_3D:
        print(
            "Error: 3D plotting not available. Install matplotlib with 3D support.",
            file=sys.stderr,
        )
        return

    # Create figure and 3D axes
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # Set dark background for better contrast
    dark_bg = _parse_color("#0a0a0a") or (0.04, 0.04, 0.04)
    fig.patch.set_facecolor(dark_bg)
    ax.set_facecolor(dark_bg)

    # Color function based on level - using def instead of lambda
    cmap = get_cmap_safe(color_scheme)

    def color_func(level: int) -> Tuple[float, float, float, float]:
        """Returns color for a given level based on colormap."""
        return cmap(level / max(max_level, 1))

    # Create the fractal
    center = np.array([0.0, 0.0, 0.0])
    _create_recursive_3d_fractal(
        ax,
        center,
        base_radius,
        0,
        max_level,
        num_children,
        scale_factor,
        min_radius,
        color_func,
        alpha_decay,
    )

    # Set plot limits
    max_extent = base_radius * (1 + 2 * scale_factor * max_level)
    ax.set_xlim([-max_extent, max_extent])
    ax.set_ylim([-max_extent, max_extent])
    ax.set_zlim([-max_extent, max_extent])

    # Configure view
    ax.view_init(elev=elev, azim=azim)

    # Remove axis ticks and labels for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # Add title
    plt.title(
        f"3D Keçeci Fractal (Levels: {max_level}, Children: {num_children})",
        color="white",
        fontsize=14,
        pad=20,
    )

    # Add lighting effect (simulated with grid)
    ax.grid(True, alpha=0.1, linestyle="--", linewidth=0.5)

    # Output handling
    output_mode = output_mode.lower().strip()
    if output_mode == "show":
        plt.show()
    elif output_mode in ["png", "jpg", "jpeg", "svg"]:
        output_filename = f"{filename}.{output_mode}"
        try:
            save_kwargs = {
                "bbox_inches": "tight",
                "pad_inches": 0.1,
                "facecolor": fig.get_facecolor(),
            }
            if output_mode in ["png", "jpg", "jpeg"]:
                save_kwargs["dpi"] = dpi
            plt.savefig(output_filename, format=output_mode, **save_kwargs)
            print(
                f"3D Fractal successfully saved to: '{os.path.abspath(output_filename)}'"
            )
        except Exception as e:
            print(
                f"Error: Could not save file '{output_filename}': {e}", file=sys.stderr
            )
        finally:
            plt.close(fig)
    else:
        print(
            f"Error: Invalid output_mode '{output_mode}'. Choose 'show', 'png', 'jpg', or 'svg'.",
            file=sys.stderr,
        )
        plt.close(fig)


# ==============================================================================
# PART 4: STRATUM MODEL VISUALIZATION
# ==============================================================================


def _draw_recursive_stratum_circles(
    ax,
    cx,
    cy,
    radius,
    level,
    max_level,
    state_collection,
    branching_rule_func,
    node_properties_func,
):
    """
    Internal recursive helper to draw the Stratum Circular Fractal.
    It uses provided functions for branching and node properties. Not for direct use.
    """
    if level >= max_level:
        return

    # Draw the main circle representing the quantum state
    level_color = plt.cm.plasma(level / max_level)
    ax.add_patch(
        plt.Circle((cx, cy), radius, facecolor=level_color, alpha=0.2, zorder=level)
    )

    # Get node properties using the PASSED-IN function
    node_props = node_properties_func(level, 0)
    ax.plot(
        cx,
        cy,
        "o",
        markersize=node_props.get("size", 10),
        color="white",
        alpha=0.8,
        zorder=level + max_level,
    )

    # Add this state's data to our collection
    state_collection.append(
        {
            "id": len(state_collection),
            "level": level,
            "energy": node_props.get("energy", 0.0),
            "size": node_props.get("size", 10),
            "color": level_color,
        }
    )

    # Determine the number of child states using the PASSED-IN function
    num_children = branching_rule_func(level)

    # Position and draw the child circles
    scale_factor = 0.5
    child_radius = radius * scale_factor
    distance_from_center = radius * (1 - scale_factor)

    for i in range(num_children):
        angle = 2 * math.pi * i / num_children + random.uniform(-0.1, 0.1)
        child_cx = cx + distance_from_center * math.cos(angle)
        child_cy = cy + distance_from_center * math.sin(angle)

        _draw_recursive_stratum_circles(
            ax,
            child_cx,
            child_cy,
            child_radius,
            level + 1,
            max_level,
            state_collection,
            branching_rule_func,
            node_properties_func,
        )


def visualize_stratum_model(
    ax,
    max_level,
    branching_rule_func,
    node_properties_func,
    initial_radius=100,
    start_cx=0,
    start_cy=0,
):
    """
    Public-facing function to visualize the Stratum Model as a circular fractal.
    This is the main entry point from your script.

    Args:
        ax: The matplotlib axes object to draw on.
        max_level (int): The maximum recursion depth.
        branching_rule_func (function): A function that takes a level (int) and returns the number of branches.
        node_properties_func (function): A function that takes a level and branch_index and returns a dict of properties (e.g., {'size': ..., 'energy': ...}).
        initial_radius (float): The radius of the first circle.
        start_cx, start_cy (float): The center coordinates of the first circle.

    Returns:
        list: A list of dictionaries, where each dictionary represents a generated state.
    """
    state_collection = []
    _draw_recursive_stratum_circles(
        ax,
        start_cx,
        start_cy,
        initial_radius,
        0,
        max_level,
        state_collection,
        branching_rule_func,
        node_properties_func,
    )
    return state_collection


def visualize_sequential_spectrum(ax, state_collection):
    """
    Draws all collected quantum states in a sequential spectrum using the Keçeci Layout,
    including dotted lines to show the connection between consecutive states.
    """
    if not state_collection:
        ax.text(0.5, 0.5, "No Data Available", color="white", ha="center", va="center")
        return

    G = nx.Graph()
    for state_data in state_collection:
        G.add_node(state_data["id"], **state_data)

    if len(G.nodes()) > 1:
        for i in range(len(G.nodes()) - 1):
            G.add_edge(i, i + 1)

    pos = kl.kececi_layout(
        G, primary_direction="top_down", primary_spacing=1.5, secondary_spacing=1.0
    )

    node_ids = list(G.nodes())
    node_sizes = [G.nodes[n].get("size", 10) * 5 for n in node_ids]
    node_colors = [G.nodes[n].get("color", "blue") for n in node_ids]

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=node_sizes,
        node_color=node_colors,
        edgecolors="white",
        linewidths=0.5,
        ax=ax,
    )

    nx.draw_networkx_edges(G, pos, ax=ax, style="dotted", edge_color="gray", alpha=0.7)

    ax.set_title(
        "Sequential State Spectrum (Keçeci Layout)", color="white", fontsize=12
    )
    ax.set_facecolor("#1a1a1a")
    ax.axis("off")


def create_color_function(
    cmap_name: str, max_level: int
) -> Callable[[int], Tuple[float, float, float, float]]:
    """
    Creates a color function that returns colors based on level.

    Args:
        cmap_name: Name of the matplotlib colormap
        max_level: Maximum level for normalization

    Returns:
        Function that takes a level and returns RGBA color
    """
    cmap = get_cmap_safe(cmap_name)

    def color_func(level: int) -> Tuple[float, float, float, float]:
        """Returns color for a given level based on colormap."""
        return cmap(level / max(max_level, 1))

    return color_func


# ==============================================================================
# PART 5: MODULE TESTS
# ==============================================================================

if __name__ == "__main__":
    # Get current script name safely
    script_name = (
        os.path.basename(sys.argv[0]) if len(sys.argv) > 0 else "kececifractals.py"
    )
    print(f"--- Running Test Cases for {script_name} ---")

    # --- General-Purpose Fractal Tests ---
    print("\n--- PART 1: General-Purpose Fractal Tests ---")
    print("\n[Test 1.1: Displaying fractal on screen (show)]")
    kececifractals_circle(
        initial_children=5,
        recursive_children=4,
        text="Keçeci Fractals",
        max_level=3,
        output_mode="show",
    )

    print("\n[Test 1.2: Saving fractal as PNG]")
    kececifractals_circle(
        initial_children=7,
        recursive_children=3,
        text="Test PNG Save",
        background_color="#101030",  # Now accepts hex strings!
        initial_circle_color="yellow",  # Now accepts color names!
        output_mode="png",
        filename="test_fractal_generic",
    )

    # --- QEC Visualization Tests ---
    print("\n--- PART 2: QEC Visualization Tests ---")
    print("\n[Test 2.1: Generating an error-free 7-qubit code...]")
    visualize_qec_fractal(
        physical_qubits_per_level=7,
        recursion_level=1,
        error_qubits=[],
        filename="QEC_Model_Test_No_Errors",
    )

    print("\n[Test 2.2: Generating a 7-qubit code with a single error...]")
    visualize_qec_fractal(
        physical_qubits_per_level=7,
        recursion_level=1,
        error_qubits=[[3]],
        filename="QEC_Model_Test_Single_Error",
    )

    print("\n[Test 2.3: Generating a 2-level code with a deep-level error...]")
    visualize_qec_fractal(
        physical_qubits_per_level=5,
        recursion_level=2,
        error_qubits=[[4, 1]],
        filename="QEC_Model_Test_Deep_Error",
    )

    # --- 3D Fractal Tests ---
    if HAS_3D:
        print("\n--- PART 3: 3D Keçeci Fractal Tests ---")
        print("\n[Test 3.1: Generating basic 3D fractal...]")
        kececifractals_3d(
            num_children=6,
            max_level=3,
            output_mode="png",
            filename="test_3d_fractal_basic",
        )

        print("\n[Test 3.2: Generating complex 3D fractal...]")
        kececifractals_3d(
            num_children=12,
            max_level=4,
            scale_factor=0.35,
            color_scheme="viridis",
            elev=25,
            azim=60,
            output_mode="png",
            filename="test_3d_fractal_complex",
        )
    else:
        print("\n--- PART 3: 3D Keçeci Fractal Tests (Skipped - 3D not available) ---")

    print("\n--- All Tests Completed ---")
