
# Keçeci Fractals: Keçeci Fraktalları (Keçeci Circle Fractal (KCF): Keçeci Dairesel Fraktalı (KDF), Keçeci-style circle fractal)

## Keçeci Fractals: Keçeci Fraktalları <img src="https://github.com/WhiteSymmetry/kececifractals/blob/main/docs/logo.jpg" alt="Keçeci Fractals: Keçeci Fraktalları" align="right" height="140"/>

[![PyPI version](https://badge.fury.io/py/kececifractals.svg)](https://badge.fury.io/py/kececifractals)
[![License: AGPL](https://img.shields.io/badge/License-AGPL-yellow.svg)](https://opensource.org/licenses/AGPL)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15392518.svg)](https://doi.org/10.5281/zenodo.15392518)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15392773.svg)](https://doi.org/10.5281/zenodo.15392773)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15396198.svg)](https://doi.org/10.5281/zenodo.15396198)

[![WorkflowHub DOI](https://img.shields.io/badge/DOI-10.48546%2Fworkflowhub.datafile.16.2-blue)](https://doi.org/10.48546/workflowhub.datafile.16.3)

[![Authorea DOI](https://img.shields.io/badge/DOI-10.22541/au.175131225.56823239/v1-blue)](https://doi.org/10.22541/au.175131225.56823239/v1)

[![Anaconda-Server Badge](https://anaconda.org/bilgi/kececifractals/badges/version.svg)](https://anaconda.org/bilgi/kececifractals)
[![Anaconda-Server Badge](https://anaconda.org/bilgi/kececifractals/badges/latest_release_date.svg)](https://anaconda.org/bilgi/kececifractals)
[![Anaconda-Server Badge](https://anaconda.org/bilgi/kececifractals/badges/platforms.svg)](https://anaconda.org/bilgi/kececifractals)
[![Anaconda-Server Badge](https://anaconda.org/bilgi/kececifractals/badges/license.svg)](https://anaconda.org/bilgi/kececifractals)

[![Open Source](https://img.shields.io/badge/Open%20Source-Open%20Source-brightgreen.svg)](https://opensource.org/)
[![Documentation Status](https://app.readthedocs.org/projects/kececifractals/badge/?0.1.0=main)](https://kececifractals.readthedocs.io/en/latest)

[![OpenSSF Best Practices](https://www.bestpractices.dev/projects//badge)](https://www.bestpractices.dev/projects/)

[![Python CI](https://github.com/WhiteSymmetry/kececifractals/actions/workflows/python_ci.yml/badge.svg?branch=main)](https://github.com/WhiteSymmetry/kececifractals/actions/workflows/python_ci.yml)
[![codecov](https://codecov.io/gh/WhiteSymmetry/kececifractals/graph/badge.svg?token=DPI71HQGNH)](https://codecov.io/gh/WhiteSymmetry/kececifractals)
[![Documentation Status](https://readthedocs.org/projects/kececifractals/badge/?version=latest)](https://kececifractals.readthedocs.io/en/latest/)
[![Binder](https://terrarium.evidencepub.io/badge_logo.svg)](https://terrarium.evidencepub.io/v2/gh/WhiteSymmetry/kececifractals/HEAD)
[![PyPI version](https://badge.fury.io/py/kececifractals.svg)](https://badge.fury.io/py/kececifractals)
[![PyPI Downloads](https://static.pepy.tech/badge/kececifractals)](https://pepy.tech/projects/kececifractals)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)
[![Linted with Ruff](https://img.shields.io/badge/Linted%20with-Ruff-green?logo=python&logoColor=white)](https://github.com/astral-sh/ruff)

![PyPI Downloads](https://img.shields.io/pypi/dm/kececifractals?logo=pypi&label=PyPi%20downloads)
[![](https://data.jsdelivr.com/v1/package/gh/WhiteSymmetry/kececifractals/badge)](https://www.jsdelivr.com/package/gh/WhiteSymmetry/kececifractals)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/kececifractals?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/kececifractals)
[![Socket Badge](https://badge.socket.dev/pypi/package/kececifractals/0.1.5?artifact_id=tar-gz)](https://socket.dev/pypi/package/kececifractals)

---

<p align="left">
    <table>
        <tr>
            <td style="text-align: center;">PyPI</td>
            <td style="text-align: center;">
                <a href="https://pypi.org/project/kececifractals/">
                    <img src="https://badge.fury.io/py/kececifractals.svg" alt="PyPI version" height="18"/>
                </a>
            </td>
        </tr>
        <tr>
            <td style="text-align: center;">Conda</td>
            <td style="text-align: center;">
                <a href="https://anaconda.org/bilgi/kececifractals">
                    <img src="https://anaconda.org/bilgi/kececifractals/badges/version.svg" alt="conda-forge version" height="18"/>
                </a>
            </td>
        </tr>
        <tr>
            <td style="text-align: center;">DOI</td>
            <td style="text-align: center;">
                <a href="https://doi.org/10.5281/zenodo.15392518">
                    <img src="https://zenodo.org/badge/DOI/10.5281/zenodo.15392518.svg" alt="DOI" height="18"/>
                </a>
            </td>
        </tr>
        <tr>
            <td style="text-align: center;">License: AGPL</td>
            <td style="text-align: center;">
                <a href="https://opensource.org/licenses/AGPL">
                    <img src="https://img.shields.io/badge/License-AGPL-yellow.svg" alt="License" height="18"/>
                </a>
            </td>
        </tr>
    </table>
</p>

---

## Description / Açıklama

**Keçeci Circle Fractal: Keçeci-style circle fractal.**: 

This module provides two primary functionalities for generating Keçeci Fractals:
1.  kececifractals_circle(): Generates general-purpose, aesthetic, and randomly
    colored circular fractals.
2.  visualize_qec_fractal(): Generates fractals customized for modeling the (version >= 0.1.1)
    concept of Quantum Error Correction (QEC) codes.
3. Stratum Model Visualization (version >= 0.1.2)
4. 3D Keçeci Fractals (version >= 0.1.6)

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

[https://zenodo.org/records/](https://doi.org/10.5281/zenodo.15392518)

https://prefix.dev/channels/bilgi/packages/kececifractals

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


![Keçeci Fractals Example](https://github.com/WhiteSymmetry/kececifractals/blob/main/examples/kf-1.png?raw=true)

![Keçeci Fractals Example](https://github.com/WhiteSymmetry/kececifractals/blob/main/examples/kf-2.png?raw=true)

![Keçeci Fractals Example](https://github.com/WhiteSymmetry/kececifractals/blob/main/examples/kf-3.png?raw=true)

![Keçeci Fractals Example](https://github.com/WhiteSymmetry/kececifractals/blob/main/examples/kf-4.png?raw=true)

![Keçeci Fractals Example](https://github.com/WhiteSymmetry/kececifractals/blob/main/examples/kf-5.png?raw=true)

![Keçeci Fractals Example](https://github.com/WhiteSymmetry/kececifractals/blob/main/examples/kf-6.png?raw=true)

![Keçeci Fractals Example](https://github.com/WhiteSymmetry/kececifractals/blob/main/examples/kf-7.png?raw=true)

---


## License / Lisans

This project is licensed under the AGPL License. See the `LICENSE` file for details.

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
Keçeci, M. (2026). Keçeci Fraktalları ile Dalga Saçılmasının Modellenmesi. Zenodo. https://doi.org/10.5281/zenodo.18359915

Keçeci, M. (2026, Ocak 24). Keçeci Fraktalları ile Dalga Saçılmasının Modellenmesi. Zenodo. https://doi.org/10.5281/zenodo.18359915

Keçeci, M. (2025). Scalable Complexity in Fractal Geometry: The Keçeci Fractal Approach. Authorea. June, 2025. https://doi.org/10.22541/au.175131225.56823239/v1

Keçeci, M. (2025). Keçeci Fractals. Open Work Flow Articles (OWFAs), WorkflowHub https://doi.org/10.48546/workflowhub.document.32.2

Keçeci, M. (2025, May 13). Scalable complexity: Mathematical analysis and potential for physical applications of the Keçeci circle fractal. Open Science Articles (OSAs), Zenodo. https://doi.org/10.5281/zenodo.15392772

Keçeci, M. (2025). kececifractals [Data set]. Open Work Flow Articles (OWFAs), WorkflowHub https://doi.org/10.48546/workflowhub.datafile.16.3

Keçeci, M. (2025, May 13). Kececifractals. Open Science Articles (OSAs), Zenodo. https://doi.org/10.5281/zenodo.15392518

```

### Chicago
```
Keçeci, Mehmet. Scalable Complexity in Fractal Geometry: The Keçeci Fractal Approach. Authorea. June, 2025. https://doi.org/10.22541/au.175131225.56823239/v1

Keçeci, Mehmet. "kececifractals" [Data set]. WorkflowHub, 2025. https://doi.org/10.48546/workflowhub.datafile.16.3

Keçeci, Mehmet. "kececifractals". Zenodo, 01 May 2025. https://doi.org/10.5281/zenodo.15392518

Keçeci, Mehmet. "Scalable Complexity: Mathematical Analysis and Potential for Physical Applications of the Keçeci Circle Fractal", 13 Mayıs 2025. https://doi.org/10.5281/zenodo.15396198.

```

🧬🦠🧊🧂💧🛢️🌀🔄➗📏🧱⚙️🎛️🌍🧵🔧🛠️🏗️🧪✅🔬🎚️🌐

Analogy: Scalable complexity: Mathematical analysis and potential for physical applications of the Keçeci circle fractal. Mehmet Keçeci

The Keçeci Circle Fractal is like an architectural system where the builder doesn’t just lay bricks—but can dynamically adjust each brick’s size, count, and spacing on demand.
While classical deterministic fractals (e.g., the Apollonian gasket) follow a rigid, unchangeable recipe—like baking a cake with a fixed mold—the KCF functions more like a parametric LEGO® kit:
every circle’s placement is governed not by inevitability, but by designer-defined parameters (scale_factor, initial_children, recursive_children).
This tunability enables faithful emulation—not just of the aesthetic self-similarity in nature, but of functionally critical traits: pore connectivity in rocks, cell-cell spacing in tissues, or bubble-wall thickness in foams—making KCF a bridge between idealized geometry and measurable physical behavior.
These analogies concretize the core innovation of the KCF—controlled, scalable complexity—while also highlighting both its mathematical flexibility and its potential for real-world physical applications.

🇹🇷 Analoji (Benzetim): Ölçeklenebilir Karmaşıklık: Keçeci Çember Fraktalinin Matematiksel Analizi ve Fiziksel Uygulamalardaki Potansiyeli. Mehmet Keçeci

Keçeci Çember Fraktalı (KÇF), bir mimârın yâlnızca tuğlalarla değil, aynı zamanda tuğlaların boyutunu, sayısını ve yerleşim sıklığını isteğe göre ayarlayabildiği bir yapı sistemine benzer:
Geleneksel fraktaller sâbit bir “yapı târifi”ne (örneğin Apollonian pastası) bağlı kalırken, Keçeci Fraktalı bir modüler yapı setidir — bir tür “Lego sistemi” gibidir:
Her parça (çember) ebeveyninin ölçeğine ve konumuna göre değil, tasarımcının seçtiği kurallara göre konumlanır.
Bu sâyede doğadaki gözenekli kayalar, köpükler veya hücre kümeleri gibi yapıların yâlnızca “güzelliğini” değil, fiziksel işlevlerini etkileyen gözenek dağılımı, yoğunluk ve yüzey pürüzlülüğü gibi kritik özelliklerini de hassasça taklit etmek mümkün hâle gelir.
Bu analojiler, KÇF’nin öne çıkan yeniliği olan kontrollü, ölçeklenebilir karmaşıklık fikrini somutlaştırır; ayrıca hem matematiksel esnekliği hem de fiziksel uygulamalara açılabilen kapısını vurgular.

![Keçeci Fractals Example](https://github.com/WhiteSymmetry/kececifractals/blob/main/examples/KCF-1.jpg?raw=true)

---

🌊🔌🌡️

Physics Analogy: Scalable complexity: Mathematical analysis and potential for physical applications of the Keçeci circle fractal. Mehmet Keçeci

The Keçeci Circle Fractal is like an acoustic diffuser panel in a recording studio:
A flat wall (simple geometry) reflects sound waves coherently—causing echoes and standing waves.
In contrast, the KCF—through its parametrically designed, multi-scale hierarchy of cavities (circles within circles)—scatters, absorbs, and redirects waves across length scales.
Just as the size, depth, and spacing of wells in a quadratic-residue diffuser determine its frequency response, the KCF’s parameters (scale_factor, recursive_children) allow predictive tuning of physical responses: thermal conductivity, electrical percolation thresholds, or electromagnetic scattering cross-sections.
Thus, the KCF is not merely a pattern—it is a geometric encoding of physical behavior.

🇹🇷 Fizik Analoji (Benzetimi): Ölçeklenebilir Karmaşıklık: Keçeci Çember Fraktalinin Matematiksel Analizi ve Fiziksel Uygulamalardaki Potansiyeli. Mehmet Keçeci

Keçeci Çember Fraktalı, bir akustik stüdyo duvarı gibidir:
Düz bir beton yüzey (basit geometri) ses dalgalarını tek bir yönde yansıtır ve yankı oluşturur;
Oysa KÇF, parametrik olarak tasarlanmış girinti-çıkıntılarla (çemberlerin hiyerarşik yerleşimi) ses dalgalarını çoklu ölçeklerde dağıtır, emer ve yönlendirir.
Tıpkı bir stüdyoda yutucu panoların boyutu, derinliği ve dağılımı sesin frekans tepkisini belirlediği gibi, KÇF’de scale_factor ve recursive_children parametreleri, bir malzemenin ısı iletimi, elektriksel direnç veya dalga saçılımı gibi fiziksel yanıtlarını hesaplanabilir ve ayarlanabilir hâle getirir.
Yâni KÇF, yâlnızca bir şekil değil — bir fiziksel davranışın geometrik kodudur.

![Keçeci Fractals Example](https://github.com/WhiteSymmetry/kececifractals/blob/main/examples/KCF-2.jpg?raw=true)

---

# Pixi:

[![Pixi](https://img.shields.io/badge/Pixi-Pixi-brightgreen.svg)](https://prefix.dev/channels/bilgi)

pixi init kececifractals

cd kececifractals

pixi workspace channel add https://repo.prefix.dev/bilgi --prepend

✔ Added https://repo.prefix.dev/bilgi

pixi add kececifractals

✔ Added kececifractals >=0.1.6,<2

pixi install

pixi shell

pixi run python -c "import kececifractals; print(kececifractals.__version__)"

### Çıktı: 0.1.6

pixi remove kececifractals

conda install -c https://prefix.dev/bilgi kececifractals

pixi run python -c "import kececifractals; print(kececifractals.__version__)"

### Çıktı: 0.1.6

pixi run pip list | grep kececifractals

### kececifractals  0.1.6

pixi run pip show kececifractals

Name: kececifractals

Version: 0.1.6

Summary: Keçeci Circle Fractal: Keçeci-style circle fractal.

Home-page: https://github.com/WhiteSymmetry/kececifractals

Author: Mehmet Keçeci

Author-email: Mehmet Keçeci <...>

License: GNU AFFERO GENERAL PUBLIC LICENSE

Copyright (c) 2025-2026 Mehmet Keçeci

