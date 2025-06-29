
# Keçeci Fractals: Keçeci Fraktals

Keçeci Circle Fractal: Keçeci-style circle fractal.

[![PyPI version](https://badge.fury.io/py/kececifractals.svg)](https://badge.fury.io/py/kececifractals)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15392518.svg)](https://doi.org/10.5281/zenodo.15392518)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15392773.svg)](https://doi.org/10.5281/zenodo.15392773)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15396198.svg)](https://doi.org/10.5281/zenodo.15396198)

[![WorkflowHub DOI](https://img.shields.io/badge/DOI-10.48546%2Fworkflowhub.datafile.16.2-blue)](https://doi.org/10.48546/workflowhub.datafile.16.2)

[![Anaconda-Server Badge](https://anaconda.org/bilgi/kececifractals/badges/version.svg)](https://anaconda.org/bilgi/kececifractals)
[![Anaconda-Server Badge](https://anaconda.org/bilgi/kececifractals/badges/latest_release_date.svg)](https://anaconda.org/bilgi/kececifractals)
[![Anaconda-Server Badge](https://anaconda.org/bilgi/kececifractals/badges/platforms.svg)](https://anaconda.org/bilgi/kececifractals)
[![Anaconda-Server Badge](https://anaconda.org/bilgi/kececifractals/badges/license.svg)](https://anaconda.org/bilgi/kececifractals)

[![Open Source](https://img.shields.io/badge/Open%20Source-Open%20Source-brightgreen.svg)](https://opensource.org/)
[![Documentation Status](https://app.readthedocs.org/projects/kececifractals/badge/?0.1.0=main)](https://kececifractals.readthedocs.io/en/latest)

[![OpenSSF Best Practices](https://www.bestpractices.dev/projects//badge)](https://www.bestpractices.dev/projects/)

---

## Description / Açıklama

**Keçeci Circle Fractal: Keçeci-style circle fractal.**: 

This module provides two primary functionalities for generating Keçeci Fractals:
1.  kececifractals_circle(): Generates general-purpose, aesthetic, and randomly
    colored circular fractals.
2.  visualize_qec_fractal(): Generates fractals customized for modeling the (version >= 0.1.1)
    concept of Quantum Error Correction (QEC) codes.
3. Stratum model Visualization (version >= 0.1.2)

Many systems encountered in nature and engineering exhibit complex and hierarchical geometric structures. Fractal geometry provides a powerful tool for understanding and modeling these structures. However, existing deterministic circle packing fractals, such as the Apollonian gasket, often adhere to fixed geometric rules and may fall short in accurately reflecting the diversity of observed structures. Addressing the need for greater flexibility in modeling physical and mathematical systems, this paper introduces the Keçeci Circle Fractal (KCF), a novel deterministic fractal. The KCF is generated through a recursive algorithm where a parent circle contains child circles scaled down by a specific `scale_factor` and whose number (`initial_children`, `recursive_children`) is controllable. These parameters allow for the tuning of the fractal's morphological characteristics (e.g., density, void distribution, boundary complexity) over a wide range. The primary advantage of the KCF lies in its tunable geometry, enabling more realistic modeling of diverse systems with varying structural parameters, such as porous media (for fluid flow simulations), granular material packings, foam structures, or potentially biological aggregations. Furthermore, the controllable structure of the KCF provides an ideal testbed for investigating structure-dependent physical phenomena like wave scattering, heat transfer, or electrical conductivity. Mathematically, it offers opportunities to study variations in fractal dimension and packing efficiency for different parameter values. In conclusion, the Keçeci Circle Fractal emerges as a valuable and versatile tool for generating geometries with controlled complexity and investigating structure-property relationships across multidisciplinary fields.

Doğada ve mühendislik uygulamalarında karşılaşılan birçok sistem, karmaşık ve hiyerarşik geometrik yapılar sergiler. Bu yapıları anlamak ve modellemek için fraktal geometri güçlü bir araç sunar. Ancak, Apollon contası gibi mevcut deterministik dairesel paketleme fraktalları genellikle sabit geometrik kurallara bağlıdır ve gözlemlenen yapıların çeşitliliğini tam olarak yansıtmakta yetersiz kalabilir. Bu çalışmada, fiziksel ve matematiksel sistemlerin modellenmesinde daha fazla esneklik sağlama ihtiyacından doğan yeni bir deterministik fraktal olan Keçeci Dairesel Fraktalı (KDF) tanıtılmaktadır. KDF, özyinelemeli bir algoritma ile üretilir; burada bir ana daire, belirli bir ölçek faktörü (`scale_factor`) ile küçültülmüş ve sayısı (`initial_children`, `recursive_children`) kontrol edilebilen çocuk daireleri içerir. Bu parametreler, fraktalın morfolojik özelliklerinin (yoğunluk, boşluk dağılımı, sınır karmaşıklığı vb.) geniş bir aralıkta ayarlanmasına olanak tanır. KDF'nin temel avantajı, bu ayarlanabilir geometrisi sayesinde, gözenekli ortamlar (akışkan simülasyonları için), granüler malzeme paketlemeleri, köpük yapıları veya potansiyel olarak biyolojik kümeleşmeler gibi yapısal parametreleri farklılık gösteren çeşitli sistemleri daha gerçekçi bir şekilde modelleyebilmesidir. Ayrıca, KDF'nin kontrol edilebilir yapısı, dalga saçılması, ısı transferi veya elektriksel iletkenlik gibi yapıya bağlı fiziksel olayların incelenmesi için ideal bir test ortamı sunar. Matematiksel olarak, farklı parametre değerleri için fraktal boyut değişimlerini ve paketleme verimliliğini inceleme imkanı sunar. Sonuç olarak, Keçeci Dairesel Fraktalı, kontrollü karmaşıklığa sahip geometriler üretmek ve çok disiplinli alanlarda yapı-özellik ilişkilerini araştırmak için değerli ve çok yönlü bir araç olarak öne çıkmaktadır.

---

## Installation / Kurulum

```bash
conda install bilgi::kececifractals -y

pip install kececifractals
```
https://anaconda.org/bilgi/kececifractals

https://pypi.org/project/kececifractals/

https://github.com/WhiteSymmetry/kececifractals

https://zenodo.org/records/

https://zenodo.org/records/

---

## Usage / Kullanım

### Example

```python
import kececifractals as kf
import importlib # Useful if you modify the .py file and want to reload it

# --- Example 1: Show the fractal inline ---
print("Generating fractal to show inline...")
kf.kececifractals_circle(
    initial_children=5,
    recursive_children=5,
    text="Keçeci Circle Fractal: Keçeci Dairesel Fraktalı",
    max_level=4,
    scale_factor=0.5,
    min_size_factor=0.001,
    output_mode='show'  # This will display the plot below the cell
)
print("Inline display finished.")

# --- Example 2: Save the fractal as an SVG file ---
print("\nGenerating fractal to save as SVG...")
kf.kececifractals_circle(
    initial_children=7,
    recursive_children=3,
    text="Keçeci Circle Fractal: Keçeci Dairesel Fraktalı",
    max_level=5,
    scale_factor=0.5,
    min_size_factor=0.001,
    base_radius=4.5,
    background_color=(0.95, 0.9, 0.85), # Light beige
    initial_circle_color=(0.3, 0.1, 0.1), # Dark brown
    output_mode='svg',
    filename="kececi_fractal_svg-1" # Will be saved in the notebook's directory
)
print("SVG saving finished.")

# --- Example 3: Save as PNG with high DPI ---
print("\nGenerating fractal to save as PNG...")
kf.kececifractals_circle(
    initial_children=4,
    recursive_children=6,
    text="Keçeci Circle Fractal: Keçeci Dairesel Fraktalı",
    max_level=6,            # Deeper recursion
    scale_factor=0.5,
    min_size_factor=0.001,  # Smaller details
    output_mode='png',
    filename="kececi_fractal_png-1",
    dpi=400                 # High resolution
)
print("PNG saving finished.")

print("\nGenerating fractal and saving as JPG...")
kf.kececifractals_circle(
    initial_children=5,
    recursive_children=7,
    text="Keçeci Circle Fractal: Keçeci Dairesel Fraktalı",
    max_level=5,
    scale_factor=0.5,
    min_size_factor=0.001,
    output_mode='jpg',      # Save as JPG
    filename="kececifractal_jpg-1",
    dpi=300                 # Medium resolution JPG
)
print("JPG saving finished.")

# --- If you modify kececifractals.py and want to reload it ---
# Without restarting the Jupyter kernel:
print("\nReloading the module...")
importlib.reload(kf)
print("Module reloaded. Now you can run the commands again with the updated code.")
kf.kececifractals_circle(output_mode='show', text="Keçeci Circle Fractal: Keçeci Dairesel Fraktalı")
```
---


---
![Keçeci Fractals Example](https://github.com/WhiteSymmetry/kececifractals/blob/main/examples/kf-1.png?raw=true)

![Keçeci Fractals Example](https://github.com/WhiteSymmetry/kececifractals/blob/main/examples/kf-2.png?raw=true)

![Keçeci Fractals Example](https://github.com/WhiteSymmetry/kececifractals/blob/main/examples/kf-3.png?raw=true)

![Keçeci Fractals Example](https://github.com/WhiteSymmetry/kececifractals/blob/main/examples/kf-4.png?raw=true)

![Keçeci Fractals Example](https://github.com/WhiteSymmetry/kececifractals/blob/main/examples/kf-5.png?raw=true)

![Keçeci Fractals Example](https://github.com/WhiteSymmetry/kececifractals/blob/main/examples/kf-6.png?raw=true)

![Keçeci Fractals Example](https://github.com/WhiteSymmetry/kececifractals/blob/main/examples/kf-7.png?raw=true)

---


---

## License / Lisans

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Citation

If this library was useful to you in your research, please cite us. Following the [GitHub citation standards](https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/creating-a-repository-on-github/about-citation-files), here is the recommended citation.

### BibTeX

```bibtex
@misc{kececi_2025_15392518,
  author       = {Keçeci, Mehmet},
  title        = {kececifractals},
  month        = may,
  year         = 2025,
  publisher    = {GitHub, PyPI, Anaconda, Zenodo},
  version      = {0.1.0},
  doi          = {10.5281/zenodo.15392518},
  url          = {https://doi.org/10.5281/zenodo.15392518},
}

@misc{kececi_2025_15396198,
  author       = {Keçeci, Mehmet},
  title        = {Scalable Complexity: Mathematical Analysis and
                   Potential for Physical Applications of the Keçeci
                   Circle Fractal
                  },
  month        = may,
  year         = 2025,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.15396198},
  url          = {https://doi.org/10.5281/zenodo.15396198},
}
```

### APA

```
Keçeci, M. (2025). kececifractals [Data set]. WorkflowHub. https://doi.org/10.48546/workflowhub.datafile.16.2

Keçeci, M. (2025). kececifractals. Zenodo. https://doi.org/10.5281/zenodo.15392518

Keçeci, M. (2025). Scalable Complexity: Mathematical Analysis and Potential for Physical Applications of the Keçeci Circle Fractal. https://doi.org/10.5281/zenodo.15396198


```

### Chicago
```
Keçeci, MehmetKeçeci, M. kececifractals [Data set]. WorkflowHub, 2025. https://doi.org/10.48546/workflowhub.datafile.16.2

Keçeci, Mehmet. "kececifractals". Zenodo, 01 May 2025. https://doi.org/10.5281/zenodo.15392518

Keçeci, Mehmet. "Scalable Complexity: Mathematical Analysis and Potential for Physical Applications of the Keçeci Circle Fractal", 13 Mayıs 2025. https://doi.org/10.5281/zenodo.15396198.

```
