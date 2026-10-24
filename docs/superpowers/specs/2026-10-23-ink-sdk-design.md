# ink-sdk — diseño y plan de implementación

Fecha: 2026-10-23
Estado: propuesta v2 para aprobar. Incorpora el render desde `tifxyz` pedido por el usuario. No hay código de producto todavía.

## 0. Resumen para decidir

- **Sí hay familias**, pero por contrato de entrada/salida, no por arquitectura. Para v0: surface-volume → mapa 2D (9 µm y canonical 2.4 µm) y surface-volume → 2D + 3D (Hecate).
- **Sí hacen falta adaptadores**, uno por receta, no por checkpoint. `ink_9um` y Reader v2 comparten adaptador; canonical y Hecate no.
- **Sí conviene config tipada**, pero no con Jinja ni un archivo por checkpoint: defaults de familia en código, un perfil TOML por modelo y opciones de corrida por invocación.
- **Standalone: decidido.** Se porta la lógica probada de Campaign X; no se importa `campaignx.*` en runtime.
- **v0 renderiza desde `tifxyz` + CT**: el SDK produce el surface volume con `vc_render_tifxyz` (VC3D/volume-cartographer), el render oficial del ecosistema Vesuvius, y después infiere. También acepta un surface volume ya renderizado. El render es un paso previo cacheado, no un adaptador de modelo.
- **Alcance: decidido.** Tres adaptadores, cinco perfiles, render, ROI, recibo y liveness. Sin entrenamiento ni evaluación científica.

## 1. Qué es y para qué

Un SDK en Python para correr **inferencia de tinta** sobre un segmento arbitrario con cualquier modelo registrado, devolviendo siempre el mismo tipo de resultado: un mapa 2D de probabilidad, con escala, orientación, identificador de modelo, digest del checkpoint y un recibo reproducible.

El objetivo no es reimplementar los modelos. Es estandarizar **la interfaz de entrada, la salida y la trazabilidad**, dejando que cada receta conserve su propio preprocesamiento.

```python
from ink_sdk import InkSDK

sdk = InkSDK(
    models_dir="~/.cache/ink-sdk/models",
    renderer="/opt/vc3d/AppRun",       # o $INK_SDK_RENDERER
)

# Caso A: el segmento trae tifxyz; el SDK renderiza desde el CT y después infiere.
result = sdk.infer(
    model="ink_9um@scrollprize-seed43-step75000",
    segment="/data/PHerc0139/segments/20230702185753/w043",  # x.tif, y.tif, z.tif, meta.json
    ct="/data/ct/PHerc0139/20230702185753.zarr",               # OME-Zarr uint8, nivel 0
    roi=(0, 2048, 0, 2048),          # y0, y1, x0, x1 en píxeles del mapa de salida
    device="auto",
    out="runs/0139-w043",
)
result.primary_map.array      # float32 HxW en [0, 1]
result.primary_map.pixel_um   # (y_um, x_um) a la escala del modelo, no la del render
result.receipt_path           # JSON con render + inferencia

# Caso B: ya hay un surface volume renderizado; no se toca el CT.
result = sdk.infer(
    model="canonical_2um@r152-3ddec-v2-l5-epoch13",
    segment="/data/PHerc0139/segments/20230702185753/w043",
    surface_volume="surface-volumes/20230702185753-2.399um.zarr",
    out="runs/0139-w043-canonical",
)
```

CLI equivalente:

```bash
ink-sdk models
ink-sdk doctor                         # verifica renderer, CT, villa, pesos
ink-sdk render ... --tifxyz ./w043 --ct ./ct.zarr --profile ink_9um@scrollprize-seed43-step75000
ink-sdk infer  ink_9um@scrollprize-seed43-step75000 --tifxyz ./w043 --ct ./ct.zarr --out runs/w043
ink-sdk infer  canonical_2um@r152-3ddec-v2-l5-epoch13 --surface-volume ./vol.zarr --out runs/w043-canonical
```

`ink-sdk render` solo produce el surface volume y su `RENDER_RECEIPT.json`; no infiere. `infer` reusa ese render por hash. No hay un “modelo de render”.

## 2. Alcance

### v0 (lo que se construye)

- Inferencia únicamente. Sin entrenamiento, sin fine-tuning, sin evaluación científica.
- **Render desde `tifxyz` + CT local** con `vc_render_tifxyz`, o uso de un **surface volume** ya renderizado en un formato y escala declarados: Zarr, `.npy` ZYX o un directorio de TIFFs numerados.
- El CT de entrada es un OME-Zarr `uint8` local, nivel 0; el SDK valida dtype, forma y chunks antes de renderizar. No baja CT desde S3 en v0.
- Selección de modelo por perfil versionado, no por nombre de arquitectura.
- ROI rectangular, device auto/cpu/cuda, salida determinista y recibo.
- **Tres adaptadores y cinco perfiles**:
  - `ink9um`: `ink_9um` seed43-step75000 y Reader v2 step040000.
  - `canonical2um`: `ink_canonical_2um` r152-3ddec-v2-l5-epoch13.
  - `hecate`: 2.4 µm y 9.6 µm.

### No-objetivos de v0

- Segmentación de superficies, unrolling, spiral fitting, corrección de malla. El SDK renderiza la malla que le dan; no la arregla.
- Renderizar sin CT local o desde formatos que no sean OME-Zarr `uint8`; tampoco baja CT remoto.
- Descargar pesos automáticamente. v0 verifica y usa un cache; `ink-sdk fetch` es M5.
- Cola remota, servidor HTTP, base de datos, UI.
- Ensembles como ciudadanos de primera clase. Se agregan después como composición.
- Prometer que “cualquier modelo” corre sobre “cualquier segmento”: cada perfil declara qué entrada acepta y el SDK rechaza lo incompatible antes de cargar pesos.

## 3. Relación con Campaign X

**El SDK es standalone.** No importa `campaignx.*` en runtime. Campaign X es la referencia probada y el patrón de oro de los tests, no una dependencia.

Se portan al SDK, con atribución y tests dorados:

- el armado de argv del lane 9 µm (`run_ink_9um.py`);
- el runner del canónico 2.4 µm (`run_ink_canonical2um.py`);
- el pooling 2.4 → 9.6 (`prepare_9um_isotropic_input.py`);
- el render `tifxyz` → surface volume (`render_vc3d.py`): orientación de normales, orden de capas, auto-crop y recibo. **No** se porta el resample a 20 px/celda (`fr/fc`) del wrapper: el SDK renderiza a escala nativa y `prepare` lleva al target del perfil, que es lo que el modelo espera;
- el gate de liveness (`lane_liveness.py`);
- el formato de recibo y la verificación de digests.

**Sobre “el render de villa”**: villa (el repo Python) no expone un renderer de tinta genérico. `vesuvius.tifxyz_label_transfer.self_render_tifxyz` renderiza `tifxyz` desde CT, pero está atado a rclone/AWS y a la transferencia de labels; no produce el stack de capas que comen los modelos. El render oficial de tinta es `vc_render_tifxyz` de VC3D/volume-cartographer, el mismo que usa `pipeline/06-tinta/render_vc3d.py`. El SDK usa ese binario, pinneado y con su SHA en el recibo. Si por “render de villa” querías `self_render_tifxyz`, decímelo antes de M1; no es el camino estándar.

Lo que se usa directo, pinneado:

- `koine_machines.inference.infer` de `ScrollPrize/villa` (MIT) para 9 µm y Reader v2;
- `hecate.py` de `scrollprize/hecate` (MIT), vendorizado;
- `model_resnet3d_3d_decoder.py` de villa (MIT), vendorizado para canonical.

**Por qué no envolver los scripts de Campaign X**: el repo destino es separado y público; un SDK que importa `campaignx.framework...` no se instala solo, y su esquema de perfiles (`campaignx.ink_lane_profile.v1`) quedaría acoplado. Portar las partes chicas y probarlas contra los runners existentes es más trabajo inicial y menos deuda después.

Si el objetivo fuera tener un prototipo en dos días, envolver Campaign X sería aceptable; no es la recomendación para un SDK.

## 4. Principio rector

**Estandarizar la interfaz, nunca el preprocesamiento.**

La lección ya está documentada en este workspace: el runner genérico de Campaign X normalizaba `clip(0,200)/200` y remuestreaba la profundidad; el canónico 2.4 µm espera `/255` sin clip y capas consecutivas. El mismo checkpoint produjo un mapa con `r=0.079` contra el mapa comunitario, frente a `r=0.885` usando su propio runner.

Y no son dos recetas: son tres normalizaciones distintas.

| modelo | normalización | profundidad | escala |
|---|---|---|---|
| `ink_9um` / Reader v2 | reconstruida desde el checkpoint por `koine_machines` | 17 capas | 9.6 ± 0.5 µm |
| `canonical_2um` | `/255`, sin clip | 62 capas consecutivas | 2.399 ± 0.05 µm |
| Hecate 2.4 | `/200`, sin clip | 64 capas | 2.4 ± 2% |
| Hecate 9.6 | `/255`, sin clip | 16 capas | 9.6 ± 2% |

Por eso:

- Cada receta conserva su runner y su normalización.
- El SDK no ofrece un `preprocess(image) -> tensor` común.
- Lo común es el contrato: qué entra, qué sale, qué se registra.

## 5. Familias y modelos de v0

No conviene agrupar por arquitectura (`ResNet`, `U-Net`, `Transformer`). Conviene agrupar por **qué entrada necesita y qué salida produce**.

| familia | entrada | salida | adaptador | modelos |
|---|---|---|---|---|
| `surface-volume-2d` | surface volume isotrópico ZYX uint8, escala declarada, ventana de profundidad | mapa 2D de probabilidad | `ink9um`, `canonical2um` | `ink_9um`, Reader v2, `ink_canonical_2um` |
| `surface-volume-3d2d` | surface volume | mapa 2D **y** volumen 3D | `hecate` | Hecate 2.4, Hecate 9.6 |
| `volume-3d` | volumen 3D (surface-conditioned o CT crudo) | volumen 3D; proyección 2D opcional | `dino3d` (posterior) | `ink_3d_dino_guided`, 3D de emel_ryan |
| `render-2d` | stack aplanado / imágenes 2D | mapa 2D | `timesformer` (posterior) | TimeSformer del GP, baseline 2.5D de Kaggle |
| `ensemble` | mapas de otros modelos | mapa combinado | `ensemble` (posterior) | Reader v2 + Hecate, SUBIT-INK |

El **render no es una familia**: produce el `surface-volume` que consumen las familias existentes. No hay un “modelo de render”; hay un paso previo del pipeline, con su propio contrato y recibo.

`ink_9um` y Reader v2 son **el mismo adaptador**: Reader v2 es un checkpoint drop-in del mismo comando (`koine_machines.inference.infer`), misma arquitectura y mismo formato de config. Cambia solo el `.pth`, el digest y, si corresponde, el piso de no-tinta.

`canonical_2um` comparte familia de entrada/salida con `ink_9um`, pero **no** comparte adaptador: distinta escala, distinta ventana, distinta normalización y distinto loader. Forzarlos al mismo adaptador repetiría el error de `r=0.079`.

### Inventario exacto de v0

| perfil | adaptador | fuente / revisión | checkpoint | sha256 | bytes | escala | capas | piso |
|---|---|---|---|---|---|---|---|---|
| `ink_9um@scrollprize-seed43-step75000` | `ink9um` | [scrollprize/ink_9um](https://huggingface.co/scrollprize/ink_9um) `7109667e…`; villa `3ea17f54…` | `hybrid_3d2d-seed43/step-075000.pth` | `2aeaa85a35ef28d7bc7bf3e848c4a6a91385e9132710927fdba41133c4ecb28f` | 138360231 | 9.6 ± 0.5 | 17 | 0.25 |
| `ink_9um@reader-v2-step040000` | `ink9um` | [domenicor046/reader-v2](https://huggingface.co/domenicor046/reader-v2) `72633d8a…` | `reader-v2-step040000.pth` | `654ec5acec2b6c4788d9cb326e2f0b8c2c730584949a934ef775db7f6ab5d3a6` | 138382373 | 9.6 ± 0.5 | 17 | 0.0¹ |
| `canonical_2um@r152-3ddec-v2-l5-epoch13` | `canonical2um` | [scrollprize/ink_canonical_2um](https://huggingface.co/scrollprize/ink_canonical_2um) `075855bc…`; villa (pin en M2) | `r152_3ddec_v2_l5_epoch13.ckpt` | `36dd0de84b7b7aa6590184192c7415466cd8a1ba7c1e59f42c6373846373c3e0` | 1549556213 | 2.399 ± 0.05 | 62 | 0.0¹ |
| `hecate@2.4um` | `hecate` | [scrollprize/hecate](https://huggingface.co/scrollprize/hecate) `9cb86e50…` | `hecate_2.4um.pth` | `fa2de3827ebd247ff2d33df1b6f899ea79d5269036cf16765997c62f0e9988c7` | 522656918 | 2.4 ± 2% | 64 | 0.0¹ |
| `hecate@9.6um` | `hecate` | [scrollprize/hecate](https://huggingface.co/scrollprize/hecate) `9cb86e50…` | `hecate_9.6um.pth` | `809f4f10f7cb7afa19b4bee0f7d2ab31edd7e11b664f9f210cc9c17f22fcfe5d` | 522576086 | 9.6 ± 2% | 16 | 0.0¹ |

¹ “0.0” significa **no hay piso declarado**, no “la no-tinta vale 0”. El perfil lleva `floor_provenance = "not-declared"` y el recibo lo dice. El único piso documentado es el 0.25 del recipe 9 µm.

La tolerancia de Hecate es la del propio CLI (2% relativo): se traduce a `scale_tolerance_um = 0.048` para 2.4 y `0.192` para 9.6. El SDK valida antes de invocar y el CLI vuelve a validar; si discrepan, gana el CLI y el error lo dice.

Fuera de v0, pero en el mapa de familias: `ink9um-dense-native` (checkpoint derivado, no es un archivo crudo de upstream), `v8-in`, `ink_3d_dino_guided`, TimeSformer, ensembles. No se registran perfiles hasta tener pin y formato de checkpoint verificables.

## 6. Arquitectura

```
Segment  ──validate──▶  ModelProfile  ──select──▶  Adapter
   │                         │                        │
   │                         │                        ├─ prepare(input)
   │                         │                        ├─ run(checkpoint, params)
   │                         │                        └─ normalize(raw) ──▶ InkMap(s)
   │                         │
   └────────────── receipt ◀─┴──────────── RunContext (device, cache, workdir)
```

### 6.1 `Segment`

Representa lo que el usuario pasa. El resolver acepta:

- un directorio `tifxyz` (`x.tif`, `y.tif`, `z.tif`, `meta.json`) + un CT OME-Zarr `uint8` local;
- un directorio de segmento con `surface-volumes/*.zarr`;
- un `.zarr` de surface volume;
- un `.npy` ZYX;
- un directorio de TIFFs numerados (stack ZYX).

```python
@dataclass(frozen=True)
class Segment:
    id: str
    root: Path
    surface_volume: Path | None
    format: str | None                 # "zarr" | "npy" | "tiff-stack"
    surface_mesh: Path | None          # tifxyz: x.tif, y.tif, z.tif, meta.json
    ct: Path | None                    # OME-Zarr uint8 local, nivel 0
    pixel_um: tuple[float, float, float] | None   # z, y, x
    depth: int | None
    dtype: str | None
    direction: str | None              # forward / reverse / unknown
    render: RenderSpec | None          # override; si falta, se deriva del perfil
```

Reglas:

- Si hay `surface_volume`, se usa; si además hay `surface_mesh` + `ct`, el surface volume gana salvo `--render`.
- Si no hay surface volume y hay `surface_mesh` + `ct`, el SDK renderiza. Si falta el CT, falla temprano con “falta CT local para renderizar”; no inventa geometría.
- El CT debe ser OME-Zarr `uint8`, nivel 0, con `0/.zarray` legible. Se rechaza `float`, `int16` o un nivel distinto: el renderer y los modelos asumen `uint8`.
- La escala del CT se toma de `--voxel-um` o de `CT/.zattrs` (`voxel_um`/`pixel_um`). Si no está, se rechaza: no se adivina. El `meta.json` de la malla se registra, pero no se usa como escala del CT.
- Si hay varios surface volumes, se elige por escala compatible con el perfil; si hay ambigüedad, se exige `--surface-volume` y se listan los candidatos.
- Si falta `pixel_um` en un surface volume, el perfil que requiere escala se rechaza. No se adivina. Se puede declarar con `--pixel-um`, y el recibo lo marca como declarado por el usuario. Para un render, `pixel_um` sale del CT y de los parámetros del render, no de `--pixel-um`.
- `surface_mesh` **sí se usa en v0**: es la entrada del render. También se conserva en el recibo cuando se usa un surface volume ya renderizado, para trazabilidad.
- Un directorio de TIFFs es un surface volume, no un `render-2d`. `render-2d` significa un stack aplanado para modelos 2D, que no es v0.

### 6.1.1 `RenderSpec`

```python
@dataclass(frozen=True)
class RenderSpec:
    scale: float = 1.0                 # píxeles por vóxel nivel 0; 1 = nativo
    num_slices: int | None = None      # default: input.depth * pool_factor
    slice_step: int = 1
    orientation: str = "auto"          # auto | inward | outward
    umbilicus: Path | None = None      # para orientar la normal como Campaign X
    roi: tuple[int, int, int, int] | None = None  # y0,y1,x0,x1 en píxeles del mapa
    renderer: Path | None = None       # override de $INK_SDK_RENDERER
```

Reglas del render:

- Se renderiza a **escala nativa** (`scale=1`) y se poolea/resamplea después. Renderizar directo a 9.6 µm con `--scale 0.25` no es equivalente al pooling 4× con el que se entrenó el modelo.
- No se aplica el zoom `fr/fc` a 20 px/celda de `render_vc3d.py`: es una convención del pipeline viejo, no del contrato del modelo. Una malla a 40 vóxeles/celda renderizada con ese zoom daría 4.8 µm y el lane 9 µm no sabría poolearla; el SDK renderiza nativo y poolea por eje.
- `num_slices` se deriva del perfil: `input.depth * pool_factor`. Para `ink_9um` sobre un CT de 2.4 µm: 17 × 4 = 68 capas nativas; sobre un CT de 9.362 µm: 17 nativas. Para canonical: 62; Hecate 2.4: 64; Hecate 9.6 sobre 2.4: 64 → pool 4 → 16.
- `slice_step` debe ser 1 en v0. Un step distinto cambia la ventana física y no está probado contra los checkpoints.
- `orientation=auto` usa el umbilicus si se pasó; si no, usa la heurística de centroide del renderer y lo registra como `orientation_source=renderer-centroid`. Para un parche sin umbilicus, la orientación de profundidad es ambigua: se avisa y, si el perfil corre una sola dirección, se exige `--normal inward|outward`.
- La orientación de lectura (transpose/flip de filas/columnas) se porta de `orientacion()` de `render_vc3d.py`: con `z` por fila/columna y la normal adentro/afuera, decide `traspuesta`, `filas_invertidas`, `columnas_invertidas`. El recibo las registra junto con `z_por_fila`/`z_por_col`. La orientación de normales y la de lectura son decisiones distintas y se registran por separado. Sin umbilicus, `adentro/afuera` se toma de la orientación de normal del renderer y el recibo marca `reading_orientation=unverified`: el mapa es válido, pero la dirección de lectura no está verificada contra el rollo.
- El ROI se implementa recortando la malla `tifxyz` con un margen (32 px) antes de renderizar, no confiando en `--crop-*` del binario: en la versión relevada, `--crop-*` cambia el tamaño del canvas pero no el origen. El recorte del mesh conserva la escala y el offset se registra.
- `--auto-crop` se usa si el binario lo expone; si no, se renderiza el canvas completo y se recorta a la bbox válida. El recibo registra `auto_crop` y la bbox.
- El render se cachea por `sha256(mesh x/y/z + meta.json)`, `sha256(CT/0/.zarray)`, `sha256(binario/AppRun)`, `scale`, `num_slices`, `slice_step`, `orientation`, `roi` y `umbilicus`. Un cambio en cualquiera invalida el cache.
- Antes de renderizar se estima `num_slices × H × W` bytes y se compara con `--max-render-gb` (default 64) y el espacio libre. Si excede, se exige `--roi` o un límite mayor; no se llena el disco ni se publica un render parcial.

### 6.2 `ModelProfile`

Un TOML validado con Pydantic (`extra="forbid"`). Ejemplo real de v0:

```toml
schema = "ink-sdk.profile.v1"
id = "ink_9um@scrollprize-seed43-step75000"
adapter = "ink9um"
family = "surface-volume-2d"

[source]
repo = "scrollprize/ink_9um"
revision = "7109667e2607db1b90c37c8b09cb876ea7fe7bb1"
upstream = "ScrollPrize/villa"
upstream_revision = "3ea17f54a9b3d5fd1aaf73e1d2c8386dbaa9f30e"
url = "https://huggingface.co/scrollprize/ink_9um"
license = "MIT"

[checkpoint]
path = "ink_9um/hybrid_3d2d-seed43/step-075000.pth"
sha256 = "2aeaa85a35ef28d7bc7bf3e848c4a6a91385e9132710927fdba41133c4ecb28f"
bytes = 138360231

[input]
kind = "surface-volume"
scale_um = 9.6
scale_tolerance_um = 0.5
depth = 17
normalization = "from-checkpoint"   # no se restatea: la trae el .pth
# El render no se configura acá: `depth` y `scale_um` fijan la ventana,
# y el resolver deriva num_slices/pool del CT. `--render-*` es override.

[execution]
direction = "both"                  # forward | reverse | both
tile = 128
overlap = 0.5
blend = "hann"
batch = 1
amp = "bf16"
tta = false

[output]
primary = "forward"
no_ink_floor = 0.25
floor_provenance = "model-card"
```

Campos que **no** son opcionales: `id`, `adapter`, `checkpoint.sha256`, `source.revision`, `input.kind`, `input.scale_um`, `input.scale_tolerance_um`, `output.no_ink_floor`, `output.floor_provenance`. Un perfil sin digest no corre: se trata como error de configuración, no como modo experimental silencioso.

### 6.3 `Renderer` y `Adapter`

El render es un paso previo, no un adaptador. El runner hace `resolver → renderer → prepare → adapter`.

```python
class Renderer(Protocol):
    name: str

    def check(self, segment: Segment, profile: ModelProfile) -> list[Issue]: ...

    def render(self, segment: Segment, spec: RenderSpec,
               ctx: RunContext) -> RenderedVolume: ...

@dataclass(frozen=True)
class RenderedVolume:
    path: Path
    shape_zyx: tuple[int, int, int]
    pixel_um: tuple[float, float, float]
    orientation: dict          # traspuesta/filas/columnas, normal, source
    receipt: dict              # hashes, comando, binario, parámetros
```

`Renderer.check` valida sin ejecutar: CT `uint8`, malla con `x/y/z` compatibles, `num_slices` positivo, ROI dentro del canvas, binario presente. `render` escribe a un directorio temporal y publica con rename atómico; nunca deja un render parcial con nombre final.

```python
class Adapter(Protocol):
    name: str

    def check(self, segment: Segment, profile: ModelProfile) -> list[Issue]: ...

    def prepare(self, segment: Segment, profile: ModelProfile,
                ctx: RunContext) -> PreparedInput: ...

    def run(self, prepared: PreparedInput, profile: ModelProfile,
            ctx: RunContext) -> RawOutput: ...

    def to_maps(self, raw: RawOutput, prepared: PreparedInput,
                profile: ModelProfile) -> dict[str, InkMap]: ...
```

`check` no carga pesos. Devuelve errores y advertencias accionables:

- escala incompatible y sin plan de pooling/resample;
- malla/CT incompatibles, CT no local o renderer ausente;
- profundidad menor que la ventana;
- checkpoint ausente o digest distinto;
- dirección desconocida cuando el perfil la requiere;
- ROI fuera del mapa.

`run` puede ser:

- **subprocess** al runner upstream (9 µm, Hecate), registrando el comando exacto, la revisión y el intérprete;
- **in-process** para canonical, que no tiene un CLI upstream y cuyo loader es un archivo vendorizado.

El SDK no reescribe ventanas deslizantes ni blending salvo que el upstream no las exponga. Cuando lo hace (canonical), el adaptador debe tener un test dorado contra el runner de Campaign X.

### 6.4 `InkMap`, `InkVolume` e `InkResult`

```python
@dataclass(frozen=True)
class InkMap:
    array: np.ndarray              # float32, HxW, [0, 1]
    valid: np.ndarray | None       # bool HxW, si el modelo reporta cobertura
    space: str                     # "surface-plane"
    pixel_um: tuple[float, float]
    origin_px: tuple[int, int]     # ROI dentro del mapa de salida completo
    direction: str                 # forward | reverse | mean
    model_id: str
    checkpoint_sha256: str
    no_ink_floor: float
    quantisation: dict             # dtype de origen, niveles, escala
    receipt: dict

@dataclass(frozen=True)
class InkVolume:
    path: Path
    dims: tuple[str, ...]          # ("z", "y", "x")
    pixel_um: tuple[float, float, float]
    evaluated_z_interval: tuple[int, int] | None
    projection: str | None         # None si el 3D es nativo y no se proyectó
    attrs: dict

@dataclass(frozen=True)
class InkResult:
    maps: dict[str, InkMap]        # "forward", "reverse", "mean"
    primary: str                   # clave de maps
    volume: InkVolume | None
    out_dir: Path | None
    receipt_path: Path | None

    @property
    def primary_map(self) -> InkMap: return self.maps[self.primary]
```

Convenciones:

- Eje 0 = Y, eje 1 = X, igual que el upstream y que `probability.npy` de Campaign X.
- `float32`, rango `[0, 1]`, sin rescale de display. El piso de no-tinta se documenta, no se resta.
- `mean` solo existe si `direction="both"`; es el promedio simple, declarado.
- Si el modelo produce un volumen 3D, `volume` apunta al `.zarr` y `maps` conserva la salida 2D nativa. Nunca se mezclan.
- Hecate 2D es una cabeza nativa, no una proyección del 3D. Para modelos 3D-only, la proyección (`max` o `mean` sobre el intervalo evaluado) se declara en el perfil; no hay default.

### 6.5 `RunContext`

- `device`: `auto`, `cpu`, `cuda`, `mps`.
- `workdir`: temporal por corrida.
- `cache_dir`: inputs preparados y descargas.
- `renderer`: path al binario/AppRun; default `$INK_SDK_RENDERER`.
- `render_cache`: default `$INK_SDK_CACHE/render`; se puede desactivar con `--no-render-cache`.
- `render_threads`: `OMP_NUM_THREADS` para el renderer; se registra. Los tests de determinismo usan 1.
- `overwrite`: por defecto `false`; nunca pisar un mapa existente.
- `dry_run`: resuelve y muestra el plan (incluido el render), no ejecuta.
- `deterministic`: fuerza algoritmos deterministas de torch; por defecto `false`.
- `progress`: callback opcional.

### 6.6 Registro

Para v0, un diccionario explícito en `registry.py`:

```python
ADAPTERS = {
    "ink9um": Ink9umAdapter,
    "canonical2um": Canonical2umAdapter,
    "hecate": HecateAdapter,
}
```

Los perfiles se descubren en `profiles/*.toml` y en `$INK_SDK_PROFILES`. Sin entry points, sin plugins, sin marketplace. Si más adelante hay terceros, se agrega `importlib.metadata` sin cambiar el contrato.

## 7. Resolución de entrada y matriz de escalas

### 7.1 Formatos

| formato | lectura | escala |
|---|---|---|
| tifxyz + OME-Zarr CT | `x.tif`, `y.tif`, `z.tif`, `meta.json`; CT `0/.zarray` | `--voxel-um` o `CT/.zattrs`; si no, error |
| OME-Zarr | `zarr.open`, array ZYX o grupo con array `0` | `.zattrs` / `metadata.json`; si no, `--pixel-um` |
| `.npy` | `np.load(mmap_mode="r")` | `--pixel-um` obligatorio |
| TIFF stack | slices numéricos ordenados por stem | `--pixel-um` obligatorio |

Un segmento puede llegar de dos formas: (a) `tifxyz` + CT, y el SDK renderiza; (b) un surface volume ya renderizado, y el SDK lo usa directo. La segunda evita el costo del render y sirve para reproducir corridas; la primera es la que hace “cualquier segmento”.

El SDK exige `uint8` sin normalizar para los tres modelos de v0. Un `float` o un array ya normalizado se rechaza: es la diferencia entre un render y una probabilidad.

### 7.2 Matriz exacta de aceptación

| escala de origen | `ink_9um` / Reader v2 | `canonical_2um` | Hecate 2.4 | Hecate 9.6 |
|---|---|---|---|---|
| 2.399 µm | pool 4× → 9.596 ✅ | nativo ✅ | nativo ✅ | pool → 9.596 ✅ (0.04% < 2%) |
| 9.362 µm | nativo ✅ | rechaza ❌ | rechaza ❌ | rechaza ❌ (2.48% > 2%) |
| 9.596 µm | nativo ✅ | rechaza ❌ | rechaza ❌ | nativo ✅ |
| 8.640 µm | resample XY opt-in → 9.362 ⚠️ | rechaza ❌ | rechaza ❌ | rechaza ❌ |

El render no cambia esta matriz: produce la escala nativa del CT y `prepare` aplica pool/resample. Para un CT de 2.399 µm y `ink_9um`, se renderizan 68 capas (17 × 4) y se poolean 4×; para un CT de 9.362 µm, se renderizan 17 capas nativas y no se poolean. Para canonical sobre 2.399, se renderizan 62 nativas.

Reglas:

- **2.4 → 9.6**: pooling 4× por eje (XY nivel 2 de la pirámide y 4× en Z por media). Se aplica por eje porque el render nativo puede no ser isótropo si el CT no lo es; para un CT 2.4 isótropo son 4/4/4. 2.399 → 9.596. Se porta `plan_pooling`/`_pool` de `prepare_9um_isotropic_input.py`; funciona sobre un array ZYX, así que acepta Zarr, `.npy` o TIFF. El `PoolingPlan` del SDK lleva factores por eje `(y, x, z)`; el script original usa un solo `xy_factor` porque asume 2.4 isótropo. Escribe un OME-Zarr temporal y cachea por hash.
- **8.64 → 9.362**: solo con `--resample-from-um 8.64`, XY lineal, Z intacto. Es el camino que Campaign X ya midió con ~4% de costo de correlación. Sin el flag, se rechaza. Se incluye en M2; si se recorta, el error dice el porcentaje y no se estira nada.
- **9.6 → 2.4**: siempre se rechaza. No se inventa resolución.
- **Hecate 9.6**: el script exige `abs(spacing/9.6 - 1) <= 0.02`; por eso 9.596 pasa y 9.362 no. El error debe decir el porcentaje y la escala exacta.
- **Canonical**: solo 2.399 ± 0.05. La escala 9.362 se rechaza sin importar `--resample`; el runner canónico no tiene un camino de upsample.

### 7.3 Flujo de “segmento arbitrario”

1. Resolver el segmento: surface volume ya renderizado, o `tifxyz` + CT.
2. Si hay que renderizar, construir el `RenderPlan` (escala nativa, `num_slices = depth × pool_factor`, orientación, ROI, hashes) y renderizar o reusar el cache.
3. Elegir el surface volume compatible con la escala del perfil.
4. Verificar profundidad, isotropicidad y rango.
5. Calcular el plan de escala (nativo, pool, resample o rechazo).
6. Mapear el ROI de salida a la grilla de entrada.
7. Preparar el input (pool/resample) y cachearlo.
8. Ejecutar el runner de la receta.
9. Normalizar a `InkMap`, correr liveness y escribir recibo (render + inferencia).

Mensaje de error, por ejemplo:

```
No hay surface volume compatible con canonical_2um@... (espera 2.399 ± 0.05 µm).
Candidatos:
  surface-volumes/20230702185753-9.362um.zarr  (9.362 µm, 65 capas)  → incompatible: este modelo no upsampling
  surface-volumes/20230702185753-2.399um.zarr  (2.399 µm, 65 capas)  → compatible
```

Si el segmento no tiene surface volume ni CT local:

```
No hay surface volume ni CT para renderizar.
  Pasá --tifxyz y --ct (OME-Zarr uint8 local), o --surface-volume.
  El render oficial es vc_render_tifxyz; `ink-sdk doctor` dice si falta.
```

## 8. Configuración: templates, perfiles y opciones de corrida

Sí a los templates, pero **no como archivos copiados por modelo**. Tres capas:

| capa | quién la escribe | qué contiene | se versiona |
|---|---|---|---|
| defaults del adaptador | el que implementa el adaptador | contrato de familia, preprocesamiento, tiling, límites | sí, en código |
| defaults de render | el que implementa el renderer | escala nativa, `slice_step=1`, orientación, cache | sí, en código |
| perfil del modelo | el que publica/registra el modelo | checkpoint, digest, escala, ventana, piso, procedencia | sí, en `profiles/*.toml` |
| opciones de corrida | el usuario en cada corrida | segmento, ROI, device, output, overrides acotados | no, salvo `--save-run` |

El render no se configura en el perfil: `input.depth` y `input.scale_um` fijan la ventana, y el resolver deriva `num_slices` y el pool del CT. `--render-*` existe solo como override de corrida y queda en el recibo.

`ink-sdk init-profile --adapter ink9um` puede generar un TOML comentado para contribuir un modelo nuevo. Es una herramienta de desarrollo, no el camino de ejecución.

Lo que **no** se hace:

- Jinja o strings templados para “adaptar” el modelo. El adaptador es código, no texto.
- Un YAML libre por modelo con claves que el SDK ignora.
- Defaults de checkpoint inventados. Sin digest, no hay corrida.
- Un archivo de config por checkpoint. El checkpoint es un campo del perfil.
- Restatear la normalización del 9 µm: la trae el checkpoint y `koine_machines` la reconstruye. El perfil dice `from-checkpoint`.

## 9. Layout del repo

```
ink-sdk/
├── pyproject.toml
├── README.md
├── docs/superpowers/specs/2026-10-23-ink-sdk-design.md
├── src/ink_sdk/
│   ├── __init__.py
│   ├── api.py              # InkSDK.infer / inspect / models
│   ├── cli.py              # typer
│   ├── contracts.py        # Segment, InkMap, InkVolume, InkResult, Issue, RunContext
│   ├── profiles.py         # Pydantic + TOML + validación
│   ├── registry.py         # adapters y descubrimiento de perfiles
│   ├── resolver.py         # segmento -> render/surface volume/escala/ROI
│   ├── render.py           # RenderPlan, RenderReceipt, cache
│   ├── renderers/
│   │   ├── base.py
│   │   └── vc_render_tifxyz.py
│   ├── prepare.py          # pooling 2.4->9.6, resample 8.64, cache
│   ├── liveness.py         # port de lane_liveness
│   ├── runner.py           # orquestación, cache, receipts
│   ├── io.py               # zarr/tiff/npy, metadata de escala
│   ├── adapters/
│   │   ├── base.py
│   │   ├── ink9um.py
│   │   ├── canonical2um.py
│   │   ├── hecate.py
│   │   └── dummy.py
│   └── _vendor/
│       ├── hecate/hecate.py
│       └── canonical/model_resnet3d_3d_decoder.py
├── profiles/
│   ├── ink_9um-scrollprize-seed43-step75000.toml
│   ├── ink_9um-reader-v2-step040000.toml
│   ├── canonical-2um.toml
│   ├── hecate-2.4um.toml
│   └── hecate-9.6um.toml
└── tests/
    ├── fixtures/
    ├── test_profiles.py
    ├── test_resolver.py
    ├── test_render.py
    ├── test_maps.py
    ├── test_adapters_contract.py
    ├── test_cli.py
    ├── test_determinism.py
    ├── test_render_golden.py
    └── test_golden_models.py   # marcado, requiere pesos/GPU
```

## 10. Dependencias, pesos y runtime

- Python 3.11+.
- Core: `numpy`, `pydantic>=2`, `typer`, `zarr`, `tifffile`. Nada de torch en el core.
- Extras:
  - `ink-sdk[render]` → `scipy` (zoom order-1) y el binario externo `vc_render_tifxyz`/AppRun.
  - `ink-sdk[ink9um]` → `torch`, `huggingface_hub` (solo para `fetch`, M5), y un checkout de villa vía `VILLA`.
  - `ink-sdk[canonical]` → `torch` + modelo vendorizado.
  - `ink-sdk[hecate]` → `torch` + `hecate.py` vendorizado.
  - `ink-sdk[all]` → todo.
- Renderer: `$INK_SDK_RENDERER` (default `$VC3D_APPRUN` o `vc_render_tifxyz` en `PATH`). `ink-sdk doctor` ejecuta `--help` y verifica las opciones que el plan usa (`--num-slices`, `--scale`, `--auto-crop`, `--flip-normals`, `--slice-step`). Si una falta, el SDK usa el fallback documentado (flip de capas en Python, `slice_step=1`) o falla antes de renderizar. El SHA-256 del binario/AppRun va al recibo. El renderer es Linux/AppImage; en macOS `doctor` lo marca no disponible y los tests de render real se saltan, pero el contrato y los tests sintéticos corren igual.
- CT: OME-Zarr `uint8` local, nivel 0. v0 no baja CT remoto; `doctor` valida `0/.zarray`, dtype y forma.
- `koine_machines` **no está en PyPI**. v0 lo resuelve con `VILLA` apuntando a un checkout pinneado de `ScrollPrize/villa` en `merge-ink-pipelines`, y ejecuta con `sys.executable` o `$INK_SDK_PYTHON`. El recibo registra `VILLA`, la revisión y el intérprete. `ink-sdk doctor` verifica que importe antes de cargar un modelo.
- Pesos: `$INK_SDK_MODELS` (default `~/.cache/ink-sdk/models`). v0 no descarga: verifica el archivo y su digest. `ink-sdk fetch` (M5, opcional) usa `huggingface_hub`, verifica `sha256` y `bytes`, y solo mueve el archivo a su lugar cuando coincide.
- Manifiesto de pesos: se reutiliza el formato de `Campaign X/framework/registries/ink-weights-0.1.0.json` (repo, path, bytes, sha256). Los checkpoints derivados (p. ej. `dense-native`) no se aceptan como si fueran crudos: el manifiesto debe declarar `derived_from` y el comando de conversión.

## 11. Salidas, recibo y liveness

Por cada corrida:

```
out/
├── render/                    # si se usó un render (nuevo o cacheado)
│   ├── 0/                     # Zarr v2 uint8 ZYX a escala nativa
│   └── RENDER_RECEIPT.json    # malla, CT, binario, orientación, crop
├── probability.npy            # float32, forward (compatibilidad Campaign X)
├── probability_reverse.npy    # si direction=both
├── probability_mean.npy       # si direction=both
├── preview.tif                # uint8, round(255*p), solo visualización
├── ink_3d.zarr                # solo Hecate con --output-3d
└── INK_SDK_RECEIPT.json
```

Decisiones:

- `probability.npy` es **forward**, igual que el mapa publicado por Campaign X. `primary` por defecto es `forward`; `--primary mean` permite cambiarlo, y el recibo lo dice.
- `probability.npy` está a la **escala del modelo**, no a la del render. El render nativo y su `pixel_um` quedan en `render/RENDER_RECEIPT.json`; el mapa no se reescala dos veces.
- El mapa hereda la orientación de lectura de la malla (transpose/flip del render). `origin_px` es relativo al mapa completo a escala del modelo; si hubo crop de malla, el offset se registra.
- Las estadísticas y el preview se calculan por mapa, no solo sobre el mean.
- El 9 µm y Hecate escriben `uint8` upstream; el SDK convierte a `float32` dividiendo por 255 y registra `quantisation` (dtype de origen, niveles distintos, rango). No se aplica el rescale de display `(p - 0.25) / 0.5`.
- Hecate 3D: `.zarr` uint8 con attrs `axes`, `spacing_um`, `evaluated_z_interval`, `outside_evaluated_z`, `background`. El 2D es la cabeza nativa, no una proyección.

### Liveness

Se porta `assess_liveness` con sus umbrales calibrados:

- `p99 - p50 >= 0.05`;
- `std >= 0.02`;
- fracción dentro de `0.05` de `0.5` `<= 0.90`;
- `EMPTY` si no hay píxeles válidos.

Se evalúa por mapa, con `valid` si el modelo reporta cobertura; para 9 µm se usa todo el plano, porque el piso 0.25 es una salida legítima. Si el mapa primario es `EMPTY` o `DEGENERATE`:

- el CLI falla con exit **3** (el mismo que lee el fleet de Campaign X), escribe `LANE_NOT_USABLE` y conserva mapa y recibo para diagnóstico;
- `--on-degenerate warn` permite continuar;
- la API Python lanza `DegenerateMapError`.

El recibo incluye:

- `profile_id`, `adapter`, versión del SDK;
- checkpoint: path, sha256, bytes, `verified` sí/no;
- input: segmento, surface volume, escala, forma, ROI, dirección;
- render: hashes de `x/y/z.tif` y `meta.json`, hash de `CT/0/.zarray`, SHA-256 del binario/AppRun, comando exacto, `scale`, `num_slices`, `slice_step`, orientación (`source`, transpose/flip), crop, forma, `pixel_um`, `layer_step_um`, `cache_hit`;
- execution: comando exacto, revisión upstream, device, batch, tile, blend, amp, `deterministic`;
- output: forma, dtype, estadísticas (p50, p90, p99, max), liveness, quantisation;
- entorno: Python, torch, CUDA, versiones de dependencias, `VILLA`;
- `non_claims`: “un mapa de probabilidad no es OCR ni prueba de tinta/letras”.

## 12. ROI

El ROI está en **píxeles del mapa de salida**, half-open: `[y0, y1) × [x0, x1)`.

- La salida tiene **exactamente** `(y1-y0, x1-x0)` píxeles. El adaptador mapea el ROI a la grilla de entrada según el plan de escala, pad-ea a múltiplos de tile si hace falta, corre y recorta.
- Si hay render, el ROI se aplica en dos lugares: se recorta la malla `tifxyz` con un margen de 32 px antes de renderizar, y se recorta el mapa a escala del modelo después. El `render_crop` queda en el recibo.
- No se usa `--crop-*` del binario para el ROI: en la versión relevada cambia el tamaño del canvas pero no el origen. Si el binario pinneado lo corrige, un test dorado lo habilita y se documenta.
- `origin_px` es el offset del recorte dentro del mapa de salida completo.
- `pixel_um` es el tamaño de píxel de salida; para 2.4 → 9.6 es 9.596, no 2.399.
- Si el ROI cae fuera del mapa, error de validación antes de cargar pesos.
- Tests con ROI impar, ROI en el borde y ROI sobre un input pooleado 4×.

Esto evita el error científico de un mapa desplazado silenciosamente, que es peor que un error visible.

## 13. Determinismo

- **Render**: misma malla, CT, binario, parámetros y `OMP_NUM_THREADS` → mismo array `uint8` (`np.array_equal`). El test usa `OMP_NUM_THREADS=1`; si el binario no es determinista con más hilos, se registra y se compara con tolerancia 0 sobre `uint8` o se documenta el límite.
- Misma GPU, mismo batch, mismo checkpoint, adaptador in-process con `--deterministic`: se exige `np.array_equal` en el test.
- Misma GPU sin esa garantía (9 µm y Hecate corren upstream como subprocess): `np.allclose(atol=1e-3, rtol=0)`. bf16 y los kernels de CUDA no garantizan bitwise. El recibo registra `deterministic: false` o `upstream`.
- Distinta GPU: no se compara píxel a píxel; se comparan `p50`, `p99` (diferencia ≤ 0.02) y correlación de Pearson ≥ 0.99 sobre píxeles válidos.
- No se fuerza `torch.use_deterministic_algorithms` por defecto: puede romper AMP o el upstream. Se expone como opción y se registra en el recibo.
- El batch por defecto es 1. Campaign X ya midió que en una GPU de 6 GB un batch 4 vacía el mapa; el SDK no hereda ese default.

## 14. Tests

- **Schema**: todos los perfiles validan; claves desconocidas fallan; digest obligatorio; `scale_tolerance_um` > 0; `no_ink_floor` en `[0, 1)`.
- **Resolver**: segmento con un volume, con varios, sin volume, escala incompatible, ROI fuera de rango. Sin GPU.
- **Render**: CT no `uint8`, malla incompleta, `num_slices` inválido, ROI fuera del canvas, orientación ambigua, cache hit/miss. Un render sintético 64×64 reproducible no requiere GPU.
- **Mapas**: dtype, forma, rango, orientación, `no_ink_floor`, `quantisation`, recibo.
- **Contrato de adaptadores**: cada adaptador declara `input.kind`, `output.kind` y rechaza un segmento incompatible antes de cargar pesos.
- **Liveness**: mapas sintéticos ALIVE / DEGENERATE / EMPTY; el CLI devuelve 3 y deja `LANE_NOT_USABLE`.
- **CLI**: `models`, `inspect`, `--dry-run`, `--json`, `--overwrite`, exit codes.
- **Golden** (`pytest -m render` para el render; `pytest -m models` para los modelos, requiere pesos y GPU):
  - `render`: SDK vs `render_vc3d.py` sobre un crop cuya malla esté a 20 vóxeles/celda, donde el wrapper no aplica su zoom `fr/fc`; misma forma, misma orientación, `uint8` igual o correlación ≥ 0.999. Para mallas de otra resolución, el SDK no usa ese zoom: se testea el pool por eje contra el target.
  - `ink9um`: SDK vs `run_ink_9um.py` de Campaign X sobre el mismo crop 256×256; correlación ≥ 0.999; misma forma; ALIVE.
  - `canonical2um`: SDK vs `run_ink_canonical2um.py`; correlación ≥ 0.99.
  - `hecate`: SDK vs `hecate.py` upstream sobre 64×64×16 (9.6) y 256×256×64 (2.4); correlación ≥ 0.999 tras decodificar el PNG.
  - Si los pesos o el fixture no están, el test se **salta con razón**, nunca pasa en silencio.
- **Determinismo**: misma GPU con `--deterministic` → igualdad; sin él → tolerancia; cross-GPU → estadísticas + correlación.

### Fixture de control

- Segmento: **PHerc0139 w043**, el control positivo ya usado por Campaign X.
- Labels: `w043_2026011217_inklabels.tif` y `..._supervision_mask.tif`, artefacto público pinneado por `sha256`; copia de desarrollo en `../data/ink_bench/0139__w043_2026011217/`, para que el repo no dependa de esa ruta.
- Surface volumes: 2.399 µm y 9.362 µm; el manifiesto del fixture fija URI remota, `sha256`, forma y escala. Si el render 9.362 no está, se usa el 2.399 pooleado a 9.596, que el lane 9 µm acepta. El ROI de 256×256 se elige de forma determinista como la ventana con más píxeles positivos del label mask, para que no sea un número inventado.
- Render: malla sintética 64×64 (`x/y/z.tif` + `meta.json`) y CT 128³ `uint8` con hashes fijos, para CI sin GPU ni CT real. El golden real usa un crop de PHerc0139 w043 con `tifxyz` + CT; si no está, se salta con razón.
- Si el volumen remoto no está disponible, el test se salta; el manifiesto y el script de provisioning quedan en `tests/fixtures/`.

## 15. Hitos

### M0 — Esqueleto y contratos (≈2 días)

- `pyproject.toml`, layout `src/`, `Segment`, `RenderSpec`, `InkMap`, `ModelProfile`, `registry`.
- Protocols `Renderer` y `Adapter`; `DummyRenderer` y `DummyAdapter` para tests sin binario.
- CLI `models`, `inspect`, `doctor`, `render --dry-run`.
- Tests de schema, resolver, mapas y CLI.

Aceptación: `ink-sdk models` lista los cinco perfiles; `inspect` muestra el plan de render + inferencia; `pytest` pasa sin torch ni binario.

### M1 — Render (≈3–4 días)

- `VcRenderTifxyzRenderer`: argv pinneado (`--scale 1`, `--num-slices`, `--slice-step`, `--auto-crop`, `--flip-normals` si el probe lo ve), ejecución con `OMP_NUM_THREADS` registrado.
- No se porta el zoom `fr/fc` a 20 px/celda; el render nativo va a `prepare`, que poolea por eje.
- Preflight de CT (`uint8`, `0/.zarray`, forma, chunks), hashes de malla/CT/binario, snapshot post-render.
- Orientación: umbilicus si se pasa; si no, centroide del renderer; flip/reverse de capas en Python; warning explícito si es ambigua.
- Crop de malla para ROI con margen de 32 px; `RENDER_RECEIPT.json`; cache por hash; publicación atómica.
- Tests sintéticos (malla 64×64 + CT 128³) y golden contra `render_vc3d.py` sobre un crop real.

Aceptación: el render del crop de PHerc0139 w043 coincide con `render_vc3d.py` en forma, orientación y `uint8`; CT no `uint8` se rechaza antes de cargar pesos; dos renders con los mismos parámetros dan `array_equal`; cache hit no re-renderiza; ROI full vs crop coincide.

### M2 — Familia 9 µm: `ink_9um` + Reader v2 (≈3 días)

- Adaptador que arma el argv y ejecuta `koine_machines.inference.infer` con `VILLA`.
- Perfiles `ink_9um` seed43-step75000 y Reader v2.
- Pooling 2.4 → 9.6 portado; resample 8.64 opt-in.
- Salida 2D, `direction=both`, recibo y liveness.
- Golden contra `run_ink_9um.py`.

Aceptación: 9.362 nativo ALIVE; 2.399 renderizado y pooleado a 9.596 ALIVE; golden r≥0.999; Reader v2 corre cambiando solo el perfil; digest verificado; dos corridas con el mismo batch dan `allclose`.

### M3 — Canonical 2.4 µm (≈2 días)

- Adaptador in-process con el loader vendorizado y la ventana 62 capas, `/255` sin clip.
- Perfil `canonical-2um`; render nativo de 62 capas sobre CT 2.399.
- Golden contra `run_ink_canonical2um.py`.

Aceptación: 2.399 nativo ALIVE; 9.362 rechazado sin cargar pesos; 62 capas; golden r≥0.99; el recibo declara `normalization=/255, no clip`.

### M4 — Hecate (≈2–3 días)

- `hecate.py` vendorizado, 2.4 y 9.6.
- Perfiles `hecate-2.4um`, `hecate-9.6um`; render nativo o pool 4× desde 2.4.
- 2D nativo; `--output-3d` opcional; `--reverse` y `direction=both` opt-in.
- Golden contra el CLI upstream.

Aceptación: 2.4 nativo; 9.596 pasa el ±2%; 9.362 se rechaza con el porcentaje; el 3D conserva `evaluated_z_interval`; golden r≥0.999.

### M5 — Endurecido (≈2 días)

- `fetch` opcional con `huggingface_hub` y verificación de sha256/bytes.
- Cache de inputs preparados, `--dry-run`, `--overwrite`, `--json`.
- README con los cinco ejemplos, incluido `tifxyz + CT`.
- CI: lint + tests sin modelos; job manual para render golden y modelos golden.
- Revisión de licencias (villa MIT, Hecate MIT, VC3D) y procedencia de pesos.

Total: **≈14–16 días** para los tres adaptadores con render. Sin render, ≈11–12 días; recortado a 9 µm + canonical con render, ≈10–11 días.

## 16. Riesgos y mitigaciones

| riesgo | mitigación |
|---|---|
| Portar Campaign X introduce drift | tests dorados contra los runners existentes; el recibo registra revisión y comando |
| `koine_machines` no está en PyPI | `VILLA` pinneado; `doctor`; recibo con revisión; vendorizar en una segunda etapa |
| Normalización distinta por receta | el adaptador la conserva; nunca un preprocess común; tests dorados |
| “Cualquier segmento” sin CT local | fallar temprano: pedir `--tifxyz` + `--ct` o `--surface-volume`; no inventar geometría |
| Binario de render distinto entre máquinas | pin de AppRun + SHA-256; `doctor` prueba `--help`; el recibo registra el binario |
| Renderer Linux en macOS | `doctor` lo marca; tests sintéticos y contrato en macOS; render real en Linux |
| Render gigante o sin memoria | ROI por crop de malla, estimación de bytes antes de correr, fallar temprano; no publicar render parcial |
| Malla con resolución distinta a 20 vóxeles/celda | no usar el zoom `fr/fc` del wrapper; render nativo + pool por eje; test con malla sintética de paso 40 |
| Orientación de normal ambigua sin umbilicus | `auto` + warning; `--normal inward` o `outward`; 9 µm corre `both`; registrar `orientation_source` |
| `--crop-*` del renderer no mueve el origen | recortar la malla con margen, no confiar en el flag; test full vs crop |
| Cache de render viejo | clave por hash de malla, CT, binario y parámetros; invalidar sin reusar por nombre |
| 9.362 rechazado por Hecate 9.6 | mensaje con el 2.48% > 2%; opción de renderizar a 9.6, no upsample silencioso |
| Mapas vacíos que parecen válidos | liveness obligatorio; exit 3; estadísticas en recibo |
| Cuantización uint8 del 9 µm/Hecate | `quantisation` en el recibo; no se compara contra un lane float como si tuvieran la misma resolución |
| VRAM insuficiente / batch alto | batch 1 por defecto; error claro; no escribir mapa parcial |
| Orientación de profundidad desconocida | 9 µm `both`; Hecate `forward` con `--reverse`/`both` opt-in; registrar la dirección |
| Licencias de checkpoints | campo `license` obligatorio; revisar antes de publicar perfiles |

## 17. Decisiones que necesito de vos

1. **Standalone vs wrapper**: **decidido: standalone.** Se porta la lógica probada de Campaign X, sin importar `campaignx.*` en runtime.
2. **Alcance de v0**: **decidido: tres adaptadores/cinco perfiles**, con render.
3. **Render**: **decidido: v0 renderiza `tifxyz` + CT** con `vc_render_tifxyz` (VC3D/volume-cartographer), el render oficial del ecosistema Vesuvius, el mismo que usa tu pipeline. No uso `self_render_tifxyz` porque es de transferencia de labels, no produce el stack de capas. Si querías otro renderer, decímelo antes de M1.
4. **Salida primaria**: recomiendo **forward**. El modelo puede leer la pila en dos sentidos; guardamos `forward`, `reverse` y `mean`, y el principal es `forward`, que es el que ya usa tu pipeline. `--primary mean` lo cambia. Si no te importa, lo dejo así.
5. **Pesos**: recomiendo **cache local + verificación de sha256** en v0; no descarga nada. Vos bajás los `.pth` una vez a `~/.cache/ink-sdk/models`. `ink-sdk fetch` (descarga automática desde Hugging Face) queda como M5 opcional, medio día más. Si querés que lo incluya, lo agrego.
6. **8.64 µm**: recomiendo **aceptarlo solo con un flag explícito** `--resample-from-um 8.64`. Algunos CT fueron escaneados a 8.64 µm en vez de 9.36; estirarlos en XY al 9.36 cuesta ~4% de correlación. Sin el flag, el SDK rechaza con el porcentaje. Es ~20 líneas y ya está medido.

Si aceptás 4, 5 y 6 como recomiendo, arranco M0/M1. Si querés cambiar alguno, decímelo y ajusto el plan antes de escribir código.
