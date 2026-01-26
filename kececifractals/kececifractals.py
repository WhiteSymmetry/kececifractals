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
import warnings
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


class KececiFractalError(Exception):
    """Keçeci Fractals için temel exception."""

    pass


class FractalParameterError(KececiFractalError):
    """Fraktal parametre hatası."""

    pass


class ColorParseError(KececiFractalError):
    """Renk parse hatası."""

    pass


class ThreeDNotSupportedError(KececiFractalError):
    """3D desteklenmiyor hatası."""

    pass


class InvalidAxisError(KececiFractalError):
    """Geçersiz eksen hatası."""

    pass


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
def draw_sphere(
    ax,
    center,
    radius,
    color,
    alpha=0.8,
    resolution_u=20,
    resolution_v=12,
    edgecolor='k',
    linewidth=0.2,
    shade=True
):
    """
    Draw a 3D sphere using plot_surface.
    
    Backward-compatible with previous versions.
    Supports customizable resolution and styling.
    
    Parameters:
        ax: matplotlib 3D axis
        center: (x, y, z) tuple or array-like
        radius: float
        color: face color
        alpha: transparency (default: 0.8)
        resolution_u: longitudinal resolution (default: 20 → matches old behavior)
        resolution_v: latitudinal resolution (default: 12 → matches old behavior)
        edgecolor: color of mesh lines (default: 'k' for visible edges)
        linewidth: width of mesh lines (default: 0.2)
        shade: enable shading (default: True)
    """
    u = np.linspace(0, 2 * np.pi, resolution_u)
    v = np.linspace(0, np.pi, resolution_v)
    u, v = np.meshgrid(u, v)

    x = center[0] + radius * np.cos(u) * np.sin(v)
    y = center[1] + radius * np.sin(u) * np.sin(v)
    z = center[2] + radius * np.cos(v)

    ax.plot_surface(
        x, y, z,
        color=color,
        alpha=alpha,
        edgecolor=edgecolor,
        linewidth=linewidth,
        shade=shade,
        antialiased=True
    )

def get_icosahedron_vertices():
    """Return 12 normalized vertices of an icosahedron for even 3D distribution."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [-1,  phi, 0], [ 1,  phi, 0], [-1, -phi, 0], [ 1, -phi, 0],
        [0, -1,  phi], [0,  1,  phi], [0, -1, -phi], [0,  1, -phi],
        [ phi, 0, -1], [ phi, 0,  1], [-phi, 0, -1], [-phi, 0,  1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    return verts / norms

"""
def draw_3d_sphere(
    ax,
    center: Tuple[float, float, float],
    radius: float,
    color: Tuple[float, float, float],
    alpha: float = 1.0,
):
    # 3D eksen üzerine küre çizer.

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
        linewidth=0.5,
    )
"""
def draw_3d_sphere(ax, center=(0,0,0), radius=1.0, color='cyan', alpha=0.3):
    """🌀 3D KÜRE - İKİ VERSİYON BİRLEŞTİRİLDİ"""
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:12j]
    x = center[0] + radius * np.cos(u) * np.sin(v)
    y = center[1] + radius * np.sin(u) * np.sin(v)
    z = center[2] + radius * np.cos(v)
    
    if isinstance(color, (tuple, list)):
        color = color
    ax.plot_surface(x, y, z, color=color, alpha=alpha, edgecolor="none",
                   antialiased=True, shade=True, linewidth=0.5)

def draw_kececi_spiral(ax, center=(0,0,0), turns=4, radius=1.2, color='#44ff88', lw=3):
    """🌀 KEÇECİ SPİRALİ"""
    t = np.linspace(0, turns*np.pi, 120)
    r = radius + 0.25*np.sin(6*t)
    x = r * np.cos(t) * np.sin(t*0.7) + center[0]
    y = r * np.sin(t) * np.cos(t*1.3) + center[1]
    z = 0.7 * np.sin(t*2.1) + center[2]
    ax.plot(x, y, z, color=color, lw=lw, alpha=0.9)

def draw_qec_vortex(ax, center=(0,0,0), major_r=1.1, minor_r=0.4, color='gold', lw=4):
    """⚛️ QEC VORTEX"""
    phi = np.linspace(0, 2*np.pi, 80)
    R = minor_r + 0.12*np.sin(9*phi)
    x = (major_r + R*np.cos(phi)) * np.cos(phi*0.4) + center[0]
    y = (major_r + R*np.cos(phi)) * np.sin(phi*0.4) + center[1]
    z = R * np.sin(phi) + center[2]
    ax.plot(x, y, z, color=color, lw=lw, alpha=0.85)

def draw_chaotic_shells(ax, scales=[0.9, 0.6, 0.35], alpha=0.3):
    """🔥 KAOTİK KÜRELER"""
    u, v = np.mgrid[0:2*np.pi:18j, 0:np.pi:14j]
    for i, scale in enumerate(scales):
        distort = 0.18 * np.sin(7*u + i*12)
        x = scale * (1+distort) * np.cos(u) * np.sin(v)
        y = scale * (1+distort) * np.sin(u) * np.sin(v)
        z = scale * np.cos(v)
        ax.plot_surface(x, y, z, color=f'C{i}', alpha=alpha)

def draw_kececi_fractal_complete(ax, pulse_center=(0,0,0), pulse_r=0.3, frame=0):
    """🏆 TAM FRACTAL"""
    draw_kececi_spiral(ax)
    draw_qec_vortex(ax)
    draw_chaotic_shells(ax)
    u, v = np.mgrid[0:2*np.pi:22j, 0:np.pi:14j]
    pulse_rad = pulse_r + 0.15*np.sin(frame*0.3)
    x = pulse_rad*np.cos(u)*np.sin(v) + pulse_center[0]
    y = pulse_rad*np.sin(u)*np.sin(v) + pulse_center[1]
    z = pulse_rad*np.cos(v) + pulse_center[2]
    ax.plot_surface(x, y, z, color='cyan', alpha=0.75)

def kececi_3d_fractal(
    num_children: int = 8,
    max_level: int = 3,
    scale_factor: float = 0.4,
    base_radius: float = 1.0,
    min_radius: float = 0.05,
    color_scheme: str = "plasma",
    alpha_decay: float = 0.7,
    figsize: Tuple[int, int] = (12, 10),
    elev: float = 30.0,
    azim: float = 45.0,
    background_color: Union[str, Tuple[float, float, float], None] = "#0a0a0a",
    show_grid: bool = True,
    grid_alpha: float = 0.1,
    title: Optional[str] = None,
    show_axis_labels: bool = False,
    axis_label_color: str = "white",
    interactive_info: bool = False,
    return_figure: bool = True,  # YENİ: Figür döndürülsün mü?
    output_mode: str = "show",  # 'show', 'save', 'return'
    filename: str = "kececi_fractal_3d",
    dpi: int = 300,
    verbose: bool = True,
) -> Union[None, Tuple[plt.Figure, plt.Axes]]:
    """
    Generates and visualizes 3D Keçeci fractals.
    
    Parameters:
    -----------
    num_children : int
        Number of child spheres at each level (default: 8)
    max_level : int
        Maximum recursion depth (default: 3)
    scale_factor : float
        Size reduction factor for child spheres (default: 0.4)
    base_radius : float
        Radius of the central sphere (default: 1.0)
    min_radius : float
        Minimum sphere radius (stops recursion when reached) (default: 0.05)
    color_scheme : str
        Matplotlib colormap name (default: 'plasma')
    alpha_decay : float
        Alpha transparency decay factor per level (default: 0.7)
    figsize : Tuple[int, int]
        Figure size (width, height) (default: (12, 10))
    elev : float
        Elevation angle for 3D view (default: 30)
    azim : float
        Azimuth angle for 3D view (default: 45)
    background_color : str or tuple
        Background color (default: '#0a0a0a')
    show_grid : bool
        Show grid lines (default: True)
    grid_alpha : float
        Grid transparency (default: 0.1)
    title : str or None
        Custom title (auto-generated if None)
    show_axis_labels : bool
        Show X, Y, Z axis labels (default: False)
    axis_label_color : str
        Color for axis labels (default: 'white')
    interactive_info : bool
        Show interactive instructions (default: False)
    return_figure : bool
        Return (fig, ax) tuple instead of showing/saving (default: True)
    output_mode : str
        'show', 'save', or 'return' (default: 'show')
    filename : str
        Base filename for saving (default: 'kececi_fractal_3d')
    dpi : int
        DPI for saved images (default: 300)
    verbose : bool
        Print progress information (default: True)
    
    Returns:
    --------
    None or Tuple[plt.Figure, plt.Axes]
        Depending on return_figure and output_mode parameters
    """
    
    if not HAS_3D:
        if verbose:
            print("Error: 3D plotting not available. Install matplotlib with 3D support.", 
                  file=sys.stderr)
        return None if not return_figure else (None, None)
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    
    # Set background color
    bg_color = _parse_color(background_color) or (0.04, 0.04, 0.04)
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    
    # Create color function
    cmap = get_cmap_safe(color_scheme)
    
    def color_func(level: int) -> Tuple[float, float, float, float]:
        """Returns color for a given level based on colormap."""
        return cmap(level / max(max_level, 1))
    
    # Generate the fractal
    center = np.array([0.0, 0.0, 0.0])
    
    if verbose:
        print("Generating 3D fractal...")
        print(f"   • Level: {max_level}")
        print(f"   • Children: {num_children}")
        print(f"   • Color scheme: {color_scheme}")
    
    _generate_recursive_3d_fractal(
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
    max_extent = base_radius * (1 + 2 * scale_factor * max_level) * 1.2
    ax.set_xlim([-max_extent, max_extent])
    ax.set_ylim([-max_extent, max_extent])
    ax.set_zlim([-max_extent, max_extent])
    
    # Configure view
    ax.view_init(elev=elev, azim=azim)
    
    # Grid settings
    if show_grid:
        ax.grid(True, alpha=grid_alpha, linestyle="--", linewidth=0.5)
    else:
        ax.grid(False)
    
    # Axis labels
    if show_axis_labels:
        ax.set_xlabel("X", fontsize=10, labelpad=10, color=axis_label_color)
        ax.set_ylabel("Y", fontsize=10, labelpad=10, color=axis_label_color)
        ax.set_zlabel("Z", fontsize=10, labelpad=10, color=axis_label_color)
        
        ax.xaxis.label.set_color(axis_label_color)
        ax.yaxis.label.set_color(axis_label_color)
        ax.zaxis.label.set_color(axis_label_color)
        ax.tick_params(axis="x", colors=axis_label_color, labelsize=8)
        ax.tick_params(axis="y", colors=axis_label_color, labelsize=8)
        ax.tick_params(axis="z", colors=axis_label_color, labelsize=8)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
    
    # Title
    if title is None:
        title = f"3D Keçeci Fractal (Levels: {max_level}, Children: {num_children})"
    
    ax.set_title(title, fontsize=14, fontweight="bold", color="white", pad=20)
    
    # Interactive info
    if interactive_info:
        info_text = (
            "Rotate: Left click + drag\n"
            "Zoom: Mouse wheel\n"
            "Pan: Right click + drag"
        )
        fig.text(
            0.02,
            0.02,
            info_text,
            fontsize=9,
            color="white",
            bbox=dict(boxstyle="round", facecolor="black", alpha=0.7),
        )
    
    plt.tight_layout()
    
    # Output handling
    output_mode = output_mode.lower().strip()
    
    if output_mode == "show":
        plt.show()
        if return_figure:
            return fig, ax
        else:
            plt.close(fig)
            return None
    
    elif output_mode == "save":
        if filename:
            output_filename = f"{filename}.png"
            try:
                save_kwargs = {
                    "bbox_inches": "tight",
                    "pad_inches": 0.1,
                    "facecolor": fig.get_facecolor(),
                    "dpi": dpi,
                }
                plt.savefig(output_filename, **save_kwargs)
                if verbose:
                    print(f"3D Fractal saved to: '{os.path.abspath(output_filename)}'")
            except Exception as e:
                print(f"Error saving file: {e}", file=sys.stderr)
            finally:
                if not return_figure:
                    plt.close(fig)
        if return_figure:
            return fig, ax
        else:
            plt.close(fig)
            return None
    
    elif output_mode == "return" or return_figure:
        # Just return the figure without showing
        return fig, ax
    
    else:
        print(f"Invalid output_mode: '{output_mode}'. Choose 'show', 'save', or 'return'.",
              file=sys.stderr)
        plt.close(fig)
        return None

"""
def kececi_3d_fractal(
    num_children: int = 8,
    max_level: int = 3,
    scale_factor: float = 0.4,
    base_radius: float = 1.0,
    min_radius: float = 0.05,
    color_scheme: str = "plasma",
    alpha_decay: float = 0.7,
    figsize: Tuple[int, int] = (12, 10),
    elev: float = 30.0,
    azim: float = 45.0,
    background_color: Union[str, Tuple[float, float, float], None] = "#0a0a0a",
    show_grid: bool = True,
    grid_alpha: float = 0.1,
    title: Optional[str] = None,
    interactive: bool = False,  # Jupyter'da interactive=False yapıyoruz
    save_filename: Optional[str] = None,
    dpi: int = 150,
):

    #3D Keçeci fraktalı oluşturur ve görselleştirir.


    if not HAS_3D:
        print("Hata: 3D grafik desteği yok. Lütfen matplotlib 3D modülünü yükleyin.")
        return None, None

    # Figür oluştur
    fig = plt.figure(figsize=figsize, facecolor="white")
    ax = fig.add_subplot(111, projection="3d")

    # Arkaplan rengini ayarla
    bg_color = _parse_color(background_color) or (0.04, 0.04, 0.04)
    fig.patch.set_facecolor("white")
    ax.set_facecolor(bg_color)

    # Renk fonksiyonunu oluştur
    color_func = generate_color_function(color_scheme, max_level)

    # Fraktalı oluştur
    center = np.array([0.0, 0.0, 0.0])
    print(f"3D fraktal oluşturuluyor...")
    print(f"   • Seviye: {max_level}")
    print(f"   • Çocuk sayısı: {num_children}")
    print(f"   • Renk şeması: {color_scheme}")

    _generate_recursive_3d_fractal(
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

    # Grafik sınırlarını ayarla
    max_extent = base_radius * (1 + 2 * scale_factor * max_level) * 1.2
    ax.set_xlim([-max_extent, max_extent])
    ax.set_ylim([-max_extent, max_extent])
    ax.set_zlim([-max_extent, max_extent])

    # Görünüm açılarını ayarla
    ax.view_init(elev=elev, azim=azim)

    # Eksen etiketlerini ve ızgarayı ayarla
    ax.set_xlabel("X", fontsize=10, labelpad=10, color="white")
    ax.set_ylabel("Y", fontsize=10, labelpad=10, color="white")
    ax.set_zlabel("Z", fontsize=10, labelpad=10, color="white")

    # Eksen rengini ayarla
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.zaxis.label.set_color("white")
    ax.tick_params(axis="x", colors="white", labelsize=8)
    ax.tick_params(axis="y", colors="white", labelsize=8)
    ax.tick_params(axis="z", colors="white", labelsize=8)

    # Izgara ayarları
    if show_grid:
        ax.grid(True, alpha=grid_alpha, linestyle="--", linewidth=0.5)
    else:
        ax.grid(False)

    # Başlık ekle
    if title is None:
        title = f"3D Keçeci Fraktalı | Seviye: {max_level} | Çocuk: {num_children}"

    ax.set_title(title, fontsize=14, fontweight="bold", color="white", pad=20)

    # Unicode karakterleri temizleyen basit bir info text
    if interactive:
        info_text = (
            "Fare ile döndür: Sol tık + sürükle\n"
            "Yakınlaştır/Uzaklaştır: Fare tekerleği\n"
            "Kaydır: Sağ tık + sürükle"
        )
        fig.text(
            0.02,
            0.02,
            info_text,
            fontsize=9,
            color="white",
            bbox=dict(boxstyle="round", facecolor="black", alpha=0.7),
        )

    # Grafik düzenini ayarla
    plt.tight_layout()

    # Kaydetme
    if save_filename:
        try:
            # Font uyarılarını geçici olarak gizle
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                plt.savefig(
                    save_filename,
                    dpi=dpi,
                    bbox_inches="tight",
                    facecolor=fig.get_facecolor(),
                    edgecolor="none",
                )
            print(f"Fraktal kaydedildi: {save_filename}")
        except Exception as e:
            print(f"Kaydetme hatası: {e}")

    print("3D fraktal hazır!")
    return fig, ax
"""


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


def _generate_recursive_3d_fractal(
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
    Recursive function to generate 3D fractal spheres.
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
        _generate_recursive_3d_fractal(
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

    # generate figure and 3D axes
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

    # generate the fractal
    center = np.array([0.0, 0.0, 0.0])
    _generate_recursive_3d_fractal(
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


def generate_color_function(
    cmap_name: str, max_level: int
) -> Callable[[int], Tuple[float, float, float, float]]:
    """
    generates a color function that returns colors based on level.

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


def optimized_3d_fractal(
    num_children: int = 6,
    max_level: int = 3,
    resolution: int = 15,  # Düşük çözünürlük için
    show_plot: bool = True,
):
    """
    Optimize edilmiş 3D fraktal (hızlı render için).
    """
    if not HAS_3D:
        return None, None

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Basit renk fonksiyonu
    def simple_color_func(level):
        colors = [(0.8, 0.2, 0.2), (0.2, 0.8, 0.2), (0.2, 0.2, 0.8), (0.8, 0.8, 0.2)]
        return colors[level % len(colors)]

    # Optimize edilmiş küre çizimi
    def draw_sphere_fast(ax, center, radius, color, alpha=0.7):
        u = np.linspace(0, 2 * np.pi, resolution)
        v = np.linspace(0, np.pi, resolution)

        x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
        y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
        z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))

        ax.plot_surface(x, y, z, color=color, alpha=alpha, edgecolor="none", shade=True)

    # Optimize edilmiş özyineleme
    def generate_fractal_fast(
        ax, center, radius, level, max_level, num_children, scale_factor
    ):
        if level > max_level or radius < 0.05:
            return

        # Küreyi çiz
        color = simple_color_func(level)
        draw_sphere_fast(ax, center, radius, color, alpha=0.7 - level * 0.15)

        # Çocuk küreler
        child_radius = radius * scale_factor

        for i in range(num_children):
            angle = 2 * np.pi * i / num_children
            elevation = np.pi * (i % 2) / 2  # Alternatif yükseklik

            x = np.cos(angle) * np.cos(elevation)
            y = np.sin(angle) * np.cos(elevation)
            z = np.sin(elevation)

            direction = np.array([x, y, z])
            direction = direction / np.linalg.norm(direction)

            child_center = center + direction * (radius + child_radius)

            generate_fractal_fast(
                ax,
                child_center,
                child_radius,
                level + 1,
                max_level,
                num_children,
                scale_factor,
            )

    # Fraktalı oluştur
    center = np.array([0.0, 0.0, 0.0])
    generate_fractal_fast(ax, center, 1.0, 0, max_level, num_children, 0.4)

    # Görünüm ayarları
    max_extent = 1.0 * (1 + 2 * 0.4 * max_level) * 1.2
    ax.set_xlim([-max_extent, max_extent])
    ax.set_ylim([-max_extent, max_extent])
    ax.set_zlim([-max_extent, max_extent])
    ax.view_init(elev=25, azim=45)

    ax.set_facecolor("#0a0a0a")
    ax.grid(True, alpha=0.1)
    ax.set_title(
        f"Hızlı 3D Fraktal (Çözünürlük: {resolution})", color="white", fontsize=12
    )

    if show_plot:
        plt.tight_layout()
        plt.show()

    return fig, ax


# Her fraktal için ayrı figür oluştur, sonra birleştir
def generate_single_fractal(num_children, max_level, color_scheme, title):
    """Tek bir fraktal oluşturur ve surface objelerini döndürür."""
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")

    color_func = generate_color_function(color_scheme, max_level)
    center = np.array([0.0, 0.0, 0.0])

    # Fraktalı oluştur
    _generate_recursive_3d_fractal(
        ax, center, 1.0, 0, max_level, num_children, 0.4, 0.05, color_func, 0.7
    )

    # Görünüm ayarları
    max_extent = 1.0 * (1 + 2 * 0.4 * max_level) * 1.2
    ax.set_xlim([-max_extent, max_extent])
    ax.set_ylim([-max_extent, max_extent])
    ax.set_zlim([-max_extent, max_extent])
    ax.view_init(elev=25, azim=45)
    ax.set_title(title, fontsize=10, color="white", pad=10)
    ax.set_facecolor("#0a0a0a")
    ax.grid(True, alpha=0.1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # Surface objelerini topla (list() kullanarak)
    surfaces = list(ax.collections)

    plt.close(fig)
    return surfaces


def generate_fractal_directly(ax, config: dict):
    """
    Fraktalı doğrudan verilen Matplotlib ekseninde oluşturur.

    Parameters:
    -----------
    ax : matplotlib.axes._subplots.Axes3DSubplot
        3D eksen objesi
    config : dict
        Fraktal konfigürasyonu:
        - num_children: Her seviyedeki çocuk sayısı
        - max_level: Maksimum özyineleme seviyesi
        - color_scheme: Renk şeması ismi
        - title: (opsiyonel) Başlık
        - scale_factor: (opsiyonel) Ölçek faktörü, varsayılan 0.4
        - base_radius: (opsiyonel) Ana yarıçap, varsayılan 1.0
        - min_radius: (opsiyonel) Minimum yarıçap, varsayılan 0.05
        - alpha_decay: (opsiyonel) Alpha azalma faktörü, varsayılan 0.7

    Returns:
    --------
    None

    Raises:
    -------
    ThreeDNotSupportedError
        3D grafik desteği yoksa
    InvalidAxisError
        Eksen 3D değilse veya geçersizse
    FractalParameterError
        Konfigürasyon parametreleri geçersizse
    """
    # 3D desteği kontrolü
    if not HAS_3D:
        raise ThreeDNotSupportedError(
            "3D grafik desteği yok. Lütfen matplotlib'in 3D modülünü yükleyin."
        )

    # Eksen kontrolü
    try:
        # Eksenin 3D olup olmadığını kontrol et
        if not hasattr(ax, "get_proj"):
            raise InvalidAxisError("Verilen eksen 3D değil.")
    except AttributeError:
        raise InvalidAxisError("Geçersiz eksen objesi.")

    # Konfigürasyon validasyonu
    required_keys = ["num_children", "max_level", "color_scheme"]
    for key in required_keys:
        if key not in config:
            raise FractalParameterError(f"Gerekli parametre eksik: '{key}'")

    # Parametre validasyonu
    if not isinstance(config["num_children"], int) or config["num_children"] < 1:
        raise FractalParameterError("num_children pozitif bir tamsayı olmalıdır.")

    if not isinstance(config["max_level"], int) or config["max_level"] < 0:
        raise FractalParameterError("max_level negatif olmayan bir tamsayı olmalıdır.")

    # Varsayılan değerleri ayarla
    config.setdefault("scale_factor", 0.4)
    config.setdefault("base_radius", 1.0)
    config.setdefault("min_radius", 0.05)
    config.setdefault("alpha_decay", 0.7)

    # Parametre aralık kontrolü
    if not (0 < config["scale_factor"] < 1):
        raise FractalParameterError("scale_factor 0 ile 1 arasında olmalıdır.")

    if config["base_radius"] <= 0:
        raise FractalParameterError("base_radius pozitif olmalıdır.")

    if config["min_radius"] <= 0:
        raise FractalParameterError("min_radius pozitif olmalıdır.")

    if not (0 <= config["alpha_decay"] <= 1):
        raise FractalParameterError("alpha_decay 0 ile 1 arasında olmalıdır.")

    # Renk fonksiyonunu oluştur
    try:
        color_func = generate_color_function(
            config["color_scheme"], config["max_level"]
        )
    except Exception as e:
        raise FractalParameterError(f"Renk şeması oluşturulamadı: {e}")

    center = np.array([0.0, 0.0, 0.0])

    # Fraktalı oluştur
    try:
        _generate_recursive_3d_fractal(
            ax,
            center,
            config["base_radius"],
            0,
            config["max_level"],
            config["num_children"],
            config["scale_factor"],
            config["min_radius"],
            color_func,
            config["alpha_decay"],
        )
    except RecursionError:
        raise FractalParameterError(
            f"Özyineleme sınırı aşıldı. max_level değerini azaltmayı deneyin."
        )
    except Exception as e:
        raise KececiFractalError(f"Fraktal oluşturulurken hata: {e}")

    # Eksen ayarları
    max_extent = (
        config["base_radius"]
        * (1 + 2 * config["scale_factor"] * config["max_level"])
        * 1.2
    )
    ax.set_xlim([-max_extent, max_extent])
    ax.set_ylim([-max_extent, max_extent])
    ax.set_zlim([-max_extent, max_extent])
    ax.view_init(elev=25, azim=45)


def generate_simple_3d_fractal(
    ax,
    num_children: int = 6,
    max_level: int = 3,
    color_scheme: str = "viridis",
    **kwargs,
):
    """
    Basit bir 3D fraktal oluşturur.

    Parameters:
    -----------
    ax : matplotlib.axes._subplots.Axes3DSubplot
        3D eksen objesi
    num_children : int, optional
        Her seviyedeki çocuk sayısı (varsayılan: 6)
    max_level : int, optional
        Maksimum özyineleme seviyesi (varsayılan: 3)
    color_scheme : str, optional
        Renk şeması ismi (varsayılan: 'viridis')
    **kwargs : dict
        Ek parametreler:
        - scale_factor: Ölçek faktörü (varsayılan: 0.4)
        - base_radius: Ana yarıçap (varsayılan: 1.0)
        - min_radius: Minimum yarıçap (varsayılan: 0.05)
        - alpha_decay: Alpha azalma faktörü (varsayılan: 0.7)
        - elev: Görünüm eğim açısı (varsayılan: 25)
        - azim: Görünüm azimut açısı (varsayılan: 45)
        - show_grid: Izgara gösterilsin mi? (varsayılan: True)
        - grid_alpha: Izgara saydamlığı (varsayılan: 0.1)
        - background_color: Arkaplan rengi (varsayılan: '#0a0a0a')
        - title: Başlık (varsayılan: None)
        - title_size: Başlık font boyutu (varsayılan: 14)
        - title_weight: Başlık font kalınlığı (varsayılan: 'bold')
        - title_color: Başlık rengi (varsayılan: 'white')
        - title_pad: Başlık padding'i (varsayılan: 20)
        - show_axis_labels: Eksen etiketleri gösterilsin mi? (varsayılan: False)
        - xlabel: X eksen etiketi (varsayılan: 'X')
        - ylabel: Y eksen etiketi (varsayılan: 'Y')
        - zlabel: Z eksen etiketi (varsayılan: 'Z')
        - axis_label_color: Eksen etiketi rengi (varsayılan: 'white')
        - axis_label_size: Eksen etiketi boyutu (varsayılan: 10)
        - tick_color: Tick rengi (varsayılan: 'white')

    Returns:
    --------
    None

    Raises:
    -------
    ThreeDNotSupportedError
        3D grafik desteği yoksa
    InvalidAxisError
        Eksen 3D değilse veya geçersizse
    FractalParameterError
        Parametreler geçersizse
    """
    # 3D desteği kontrolü
    if not HAS_3D:
        raise ThreeDNotSupportedError(
            "3D grafik desteği yok. Lütfen matplotlib'in 3D modülünü yükleyin."
        )

    # Eksen kontrolü
    try:
        if not hasattr(ax, "get_proj"):
            raise InvalidAxisError("Verilen eksen 3D değil.")
    except AttributeError:
        raise InvalidAxisError("Geçersiz eksen objesi.")

    # Parametre validasyonu
    if not isinstance(num_children, int) or num_children < 1:
        raise FractalParameterError("num_children pozitif bir tamsayı olmalıdır.")

    if not isinstance(max_level, int) or max_level < 0:
        raise FractalParameterError("max_level negatif olmayan bir tamsayı olmalıdır.")

    # Varsayılan değerleri ayarla
    scale_factor = kwargs.get("scale_factor", 0.4)
    base_radius = kwargs.get("base_radius", 1.0)
    min_radius = kwargs.get("min_radius", 0.05)
    alpha_decay = kwargs.get("alpha_decay", 0.7)
    elev = kwargs.get("elev", 25)
    azim = kwargs.get("azim", 45)
    show_grid = kwargs.get("show_grid", True)
    grid_alpha = kwargs.get("grid_alpha", 0.1)
    background_color = kwargs.get("background_color", "#0a0a0a")
    title = kwargs.get("title", None)

    # Parametre aralık kontrolü
    if not (0 < scale_factor < 1):
        raise FractalParameterError("scale_factor 0 ile 1 arasında olmalıdır.")

    if base_radius <= 0:
        raise FractalParameterError("base_radius pozitif olmalıdır.")

    if min_radius <= 0:
        raise FractalParameterError("min_radius pozitif olmalıdır.")

    if not (0 <= alpha_decay <= 1):
        raise FractalParameterError("alpha_decay 0 ile 1 arasında olmalıdır.")

    if not (-90 <= elev <= 90):
        raise FractalParameterError("elev -90 ile 90 arasında olmalıdır.")

    if not (0 <= azim <= 360):
        raise FractalParameterError("azim 0 ile 360 arasında olmalıdır.")

    # Arkaplan rengini ayarla
    try:
        bg_color = _parse_color(background_color) or (0.04, 0.04, 0.04)
        ax.set_facecolor(bg_color)
    except Exception as e:
        raise ColorParseError(f"Arkaplan rengi parse edilemedi: {e}")

    # Renk fonksiyonunu oluştur
    try:
        color_func = generate_color_function(color_scheme, max_level)
    except Exception as e:
        raise FractalParameterError(f"Renk şeması oluşturulamadı: {e}")

    center = np.array([0.0, 0.0, 0.0])

    # Fraktalı oluştur
    try:
        _generate_recursive_3d_fractal(
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
    except RecursionError:
        raise FractalParameterError(
            f"Özyineleme sınırı aşıldı. max_level değerini azaltmayı deneyin."
        )
    except Exception as e:
        raise KececiFractalError(f"Fraktal oluşturulurken hata: {e}")

    # Eksen ayarları
    max_extent = base_radius * (1 + 2 * scale_factor * max_level) * 1.2
    ax.set_xlim([-max_extent, max_extent])
    ax.set_ylim([-max_extent, max_extent])
    ax.set_zlim([-max_extent, max_extent])
    ax.view_init(elev=elev, azim=azim)

    # Izgara ayarları
    if show_grid:
        ax.grid(True, alpha=grid_alpha, linestyle="--", linewidth=0.5)
    else:
        ax.grid(False)

    # Başlık ekle
    if title:
        ax.set_title(
            title,
            fontsize=kwargs.get("title_size", 14),
            fontweight=kwargs.get("title_weight", "bold"),
            color=kwargs.get("title_color", "white"),
            pad=kwargs.get("title_pad", 20),
        )

    # Eksen etiketleri
    if kwargs.get("show_axis_labels", False):
        ax.set_xlabel(
            kwargs.get("xlabel", "X"),
            color=kwargs.get("axis_label_color", "white"),
            fontsize=kwargs.get("axis_label_size", 10),
        )
        ax.set_ylabel(
            kwargs.get("ylabel", "Y"),
            color=kwargs.get("axis_label_color", "white"),
            fontsize=kwargs.get("axis_label_size", 10),
        )
        ax.set_zlabel(
            kwargs.get("zlabel", "Z"),
            color=kwargs.get("axis_label_color", "white"),
            fontsize=kwargs.get("axis_label_size", 10),
        )

        # Eksen etiketi renkleri
        ax.xaxis.label.set_color(kwargs.get("axis_label_color", "white"))
        ax.yaxis.label.set_color(kwargs.get("axis_label_color", "white"))
        ax.zaxis.label.set_color(kwargs.get("axis_label_color", "white"))

        # Tick renkleri
        ax.tick_params(axis="x", colors=kwargs.get("tick_color", "white"))
        ax.tick_params(axis="y", colors=kwargs.get("tick_color", "white"))
        ax.tick_params(axis="z", colors=kwargs.get("tick_color", "white"))
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])


# ==============================================================================
# ÖRNEK KULLANIM FONKSİYONLARI (isteğe bağlı) - DÜZELTİLMİŞ
# ==============================================================================


def example_multiple_fractals():
    """
    Çoklu fraktal karşılaştırması örneği.

    Returns:
    --------
    matplotlib.figure.Figure or None
        Oluşturulan figür veya hata durumunda None
    """
    if not HAS_3D:
        print("Hata: 3D grafik desteği yok.")
        return None

    try:
        import matplotlib.pyplot as plt

        # Ana figür oluştur
        fig, axes = plt.subplots(
            2, 2, figsize=(15, 12), subplot_kw={"projection": "3d"}
        )
        fig.patch.set_facecolor("#111111")

        # Farklı parametre kombinasyonları
        configs = [
            {
                "num_children": 4,
                "max_level": 2,
                "color_scheme": "viridis",
                "title": "Küçük Fraktal",
            },
            {
                "num_children": 8,
                "max_level": 3,
                "color_scheme": "plasma",
                "title": "Orta Fraktal",
            },
            {
                "num_children": 12,
                "max_level": 3,
                "color_scheme": "summer",
                "title": "Yoğun Fraktal",
            },
            {
                "num_children": 6,
                "max_level": 4,
                "color_scheme": "cool",
                "title": "Derin Fraktal",
            },
        ]

        # Her fraktalı doğrudan kendi ekseninde oluştur
        for idx, (ax, config) in enumerate(zip(axes.flat, configs)):
            try:
                generate_fractal_directly(ax, config)

                # Eksen görünüm ayarları
                ax.set_title(
                    config["title"],
                    fontsize=11,
                    fontweight="bold",
                    color="white",
                    pad=15,
                )
                ax.set_facecolor("#0a0a0a")
                ax.grid(True, alpha=0.15)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_zticks([])

            except Exception as e:
                print(f"Fraktal {config['title']} oluşturulurken hata: {e}")
                # Hata durumunda boş bir metin göster
                ax.text(0.5, 0.5, 0.5, "Hata", color="red", ha="center", va="center")

        plt.suptitle(
            "Farklı 3D Keçeci Fraktal Çeşitleri",
            fontsize=16,
            fontweight="bold",
            color="white",
            y=0.95,
        )
        plt.tight_layout()

        return fig

    except Exception as e:
        print(f"Çoklu fraktal örneği oluşturulurken hata: {e}")
        return None


def example_view_angles():
    """
    Farklı görünüm açıları örneği.

    Returns:
    --------
    matplotlib.figure.Figure or None
        Oluşturulan figür veya hata durumunda None
    """
    if not HAS_3D:
        print("Hata: 3D grafik desteği yok.")
        return None

    try:
        import matplotlib.pyplot as plt

        # Tek bir fraktal oluştur ve farklı açılardan göster
        fig = plt.figure(figsize=(12, 8))
        fig.patch.set_facecolor("#111111")

        # Fraktal parametreleri
        fractal_params = {
            "num_children": 8,
            "max_level": 3,
            "scale_factor": 0.4,
            "color_scheme": "hot",
        }

        # Tüm alt eksenlerde aynı fraktalı oluştur
        view_angles = [
            (30, 0, "Ön Görünüm"),
            (30, 90, "Sağ Görünüm"),
            (30, 180, "Arka Görünüm"),
            (30, 270, "Sol Görünüm"),
        ]

        for idx, (elev, azim, title) in enumerate(view_angles, 1):
            ax = fig.add_subplot(2, 2, idx, projection="3d")

            # Fraktalı bu eksende oluştur
            try:
                generate_simple_3d_fractal(
                    ax,
                    num_children=fractal_params["num_children"],
                    max_level=fractal_params["max_level"],
                    color_scheme=fractal_params["color_scheme"],
                    scale_factor=fractal_params["scale_factor"],
                    elev=elev,
                    azim=azim,
                    title=f"{title}\n(elev={elev}°, azim={azim}°)",
                    title_size=10,
                    show_axis_labels=False,
                )

                # Ek ayarlar
                ax.set_facecolor("#0a0a0a")
                ax.grid(True, alpha=0.1)

            except Exception as e:
                print(f"Görünüm açısı {title} oluşturulurken hata: {e}")
                ax.text(0.5, 0.5, 0.5, "Hata", color="red", ha="center", va="center")

        plt.suptitle(
            "3D Keçeci Fraktalı - Farklı Görünüm Açıları",
            fontsize=14,
            fontweight="bold",
            color="white",
            y=0.95,
        )
        plt.tight_layout()

        return fig

    except Exception as e:
        print(f"Görünüm açıları örneği oluşturulurken hata: {e}")
        return None


def example_simple_fractal():
    """
    Basit fraktal örneği.

    Returns:
    --------
    matplotlib.figure.Figure or None
        Oluşturulan figür veya hata durumunda None
    """
    if not HAS_3D:
        print("Hata: 3D grafik desteği yok.")
        return None

    try:
        import matplotlib.pyplot as plt

        # Yeni bir figür oluştur
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Fraktalı oluştur
        generate_simple_3d_fractal(
            ax,
            num_children=7,
            max_level=4,
            color_scheme="coolwarm",
            title="Basit 3D Keçeci Fraktalı",
            show_axis_labels=True,
            xlabel="X Ekseni",
            ylabel="Y Ekseni",
            zlabel="Z Ekseni",
        )

        # Figür arkaplan rengi
        fig.patch.set_facecolor("#111111")
        plt.tight_layout()

        return fig

    except Exception as e:
        print(f"Basit fraktal örneği oluşturulurken hata: {e}")
        return None


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
