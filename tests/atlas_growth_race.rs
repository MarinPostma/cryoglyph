//! Regression test for a wgpu command-ordering hazard in `InnerAtlas::grow`.
//!
//! `grow()` records a `copy_texture_to_texture` (old atlas -> new, larger
//! atlas) into the caller-supplied `CommandEncoder`. That command is
//! *deferred*: it only takes its place on the device timeline when the
//! encoder is eventually finished and submitted via `Queue::submit`.
//!
//! Meanwhile, `TextRenderer::prepare_with_depth` writes each newly
//! rasterized glyph's pixels into the atlas texture via `Queue::write_texture`
//! - an *immediate* queue operation that (per wgpu's ordering contract) is
//! guaranteed to execute before any command buffer submitted *after* the
//! `write_texture` call.
//!
//! Within a single `prepare()` invocation, several `grow()` calls (and their
//! `queue.write_texture` glyph uploads) can happen back to back, all while
//! sharing one still-unsubmitted `CommandEncoder`. Every `copy_texture_to_texture`
//! recorded during that call ends up in the same command buffer, submitted
//! once, after prepare() returns - i.e. after *every* `write_texture` call
//! made during that prepare(). So when the batched copies finally execute,
//! a later grow's "copy old atlas into new atlas" can stomp over pixels that
//! an earlier `write_texture` call already placed in the overlapping region
//! of the new, larger texture - even though that copy was *recorded* before
//! the write happened. The result: a glyph that was rasterized and uploaded
//! correctly ends up silently reverted to whatever was in that atlas slot
//! before the glyph existed (typically nothing / transparent), i.e. the
//! character renders blank.
//!
//! This test reproduces the bursty, many-distinct-glyphs-at-once access
//! pattern of a fresh app's first frame (multiple font families, multiple
//! sizes, enough distinct glyphs to force several atlas growths within one
//! `prepare()` call) and then reads back the actual rendered pixels to check
//! whether every requested, ink-bearing glyph is actually visible.
//!
//! A "control" variant is included that forces a `Queue::submit` between
//! small batches, so any grow's deferred copy always executes before the
//! next batch's `write_texture` calls happen. If the hypothesis above is
//! correct, the single-encoder burst should show blank glyphs while the
//! submit-between-batches control should not.

use cosmic_text::{Attrs, Buffer, Color as TextColor, Family, FontSystem, Metrics, Shaping};
use cryoglyph::{Cache, ColorMode, Resolution, TextArea, TextAtlas, TextBounds, TextRenderer};
use pollster::block_on;
use wgpu::{
    CommandEncoderDescriptor, Extent3d, MultisampleState, TexelCopyBufferInfo,
    TexelCopyBufferLayout, TexelCopyTextureInfo, TextureAspect, TextureDescriptor,
    TextureDimension, TextureFormat, TextureUsages, TextureViewDescriptor,
};

const CELL: u32 = 160;
const COLS: u32 = 20;
// Alphanumeric charset (no spaces / no zero-ink glyphs).
const CHARS: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
const SIZES: [f32; 3] = [36.0, 64.0, 96.0];
const FAMILIES: [Family<'static>; 2] = [Family::SansSerif, Family::Monospace];

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl Gpu {
    fn new() -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());

        let adapter = block_on(wgpu::util::initialize_adapter_from_env_or_default(
            &instance, None,
        ))
        .expect("no wgpu adapter available in this environment");

        let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("atlas-growth-race test device"),
            required_features: wgpu::Features::empty(),
            required_limits: adapter.limits(),
            memory_hints: wgpu::MemoryHints::Performance,
            ..wgpu::DeviceDescriptor::default()
        }))
        .expect("failed to create wgpu device");

        Self { device, queue }
    }
}

/// One glyph cell we asked to have rendered, and where on the canvas it lives.
struct Cell {
    label: String,
    col: u32,
    row: u32,
}

fn build_buffers(font_system: &mut FontSystem) -> (Vec<Buffer>, Vec<Cell>) {
    let mut buffers = Vec::new();
    let mut cells = Vec::new();

    let mut index = 0u32;
    for family in FAMILIES {
        for size in SIZES {
            for ch in CHARS.chars() {
                let attrs = Attrs::new().family(family);
                let metrics = Metrics::new(size, size * 1.25);
                let mut buffer = Buffer::new(font_system, metrics);
                buffer.set_size(Some(CELL as f32), Some(CELL as f32));
                buffer.set_text(&ch.to_string(), &attrs, Shaping::Advanced, None);
                buffer.shape_until_scroll(font_system, false);

                buffers.push(buffer);
                cells.push(Cell {
                    label: format!("{family:?} {size}px '{ch}'"),
                    col: index % COLS,
                    row: index / COLS,
                });
                index += 1;
            }
        }
    }

    (buffers, cells)
}

fn text_area<'a>(buffer: &'a Buffer, cell: &Cell) -> TextArea<'a> {
    let left = (cell.col * CELL) as f32;
    let top = (cell.row * CELL) as f32;
    TextArea {
        text: buffer.layout_runs(),
        left,
        top,
        scale: 1.0,
        bounds: TextBounds {
            left: left as i32,
            top: top as i32,
            right: (left as i32) + CELL as i32,
            bottom: (top as i32) + CELL as i32,
        },
        default_color: TextColor::rgb(255, 255, 255),
    }
}

/// Renders `atlas`'s currently-prepared glyphs to an offscreen target and
/// returns, per grid cell, whether any bright ("ink") pixel was found.
fn render_and_read_ink(
    gpu: &Gpu,
    atlas: &mut TextAtlas,
    renderer: &TextRenderer,
    viewport: &cryoglyph::Viewport,
    canvas_width: u32,
    canvas_height: u32,
) -> Vec<bool> {
    let target = gpu.device.create_texture(&TextureDescriptor {
        label: Some("atlas-growth-race target"),
        size: Extent3d {
            width: canvas_width,
            height: canvas_height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let target_view = target.create_view(&TextureViewDescriptor::default());

    let mut encoder = gpu
        .device
        .create_command_encoder(&CommandEncoderDescriptor {
            label: Some("atlas-growth-race render"),
        });

    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("atlas-growth-race pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &target_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        renderer
            .render(atlas, viewport, &mut pass)
            .expect("render text");
    }

    let bytes_per_pixel = 4u32;
    let unpadded_bytes_per_row = canvas_width * bytes_per_pixel;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded_bytes_per_row = unpadded_bytes_per_row.div_ceil(align) * align;

    let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("atlas-growth-race readback"),
        size: (padded_bytes_per_row * canvas_height) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    encoder.copy_texture_to_buffer(
        TexelCopyTextureInfo {
            texture: &target,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: TextureAspect::All,
        },
        TexelCopyBufferInfo {
            buffer: &readback,
            layout: TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row),
                rows_per_image: None,
            },
        },
        Extent3d {
            width: canvas_width,
            height: canvas_height,
            depth_or_array_layers: 1,
        },
    );

    gpu.queue.submit(std::iter::once(encoder.finish()));

    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    gpu.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    rx.recv().unwrap().expect("map readback buffer");

    let data = slice.get_mapped_range().to_vec();
    readback.unmap();

    let rows = canvas_height / CELL;
    let cols = canvas_width / CELL;
    let mut has_ink = vec![false; (rows * cols) as usize];

    for row in 0..rows {
        for col in 0..cols {
            let mut ink = false;
            'scan: for y in (row * CELL)..((row + 1) * CELL).min(canvas_height) {
                let row_start = (y * padded_bytes_per_row) as usize;
                for x in (col * CELL)..((col + 1) * CELL).min(canvas_width) {
                    let px = row_start + (x * bytes_per_pixel) as usize;
                    if data[px] > 128 {
                        ink = true;
                        break 'scan;
                    }
                }
            }
            has_ink[(row * cols + col) as usize] = ink;
        }
    }

    has_ink
}

/// Bursts every glyph through a single `prepare()` call, sharing one
/// still-unsubmitted encoder - exactly the pattern of a first frame that
/// requests a dense burst of distinct glyphs (prose + icon fonts + UI
/// chrome) nearly simultaneously.
#[test]
fn burst_in_single_encoder_corrupts_glyphs() {
    let gpu = Gpu::new();
    let mut font_system = FontSystem::new();
    let mut swash_cache = cryoglyph::SwashCache::new();
    let cache = Cache::new(&gpu.device);
    let mut viewport = cryoglyph::Viewport::new(&gpu.device, &cache);
    let mut atlas = TextAtlas::with_color_mode(
        &gpu.device,
        &gpu.queue,
        &cache,
        TextureFormat::Rgba8Unorm,
        ColorMode::Web,
    );
    let mut renderer = TextRenderer::new(&mut atlas, &gpu.device, MultisampleState::default(), None);

    let (buffers, cells) = build_buffers(&mut font_system);
    let canvas_width = COLS * CELL;
    let canvas_height = (cells.len() as u32).div_ceil(COLS) * CELL;
    viewport.update(
        &gpu.queue,
        Resolution {
            width: canvas_width,
            height: canvas_height,
        },
    );

    let text_areas: Vec<TextArea> = buffers
        .iter()
        .zip(cells.iter())
        .map(|(buffer, cell)| text_area(buffer, cell))
        .collect();

    let mut encoder = gpu
        .device
        .create_command_encoder(&CommandEncoderDescriptor {
            label: Some("atlas-growth-race burst"),
        });

    renderer
        .prepare(
            &gpu.device,
            &gpu.queue,
            &mut encoder,
            &mut font_system,
            &mut atlas,
            &viewport,
            text_areas,
            &mut swash_cache,
        )
        .expect("prepare should not report AtlasFull for this glyph count");

    // All of the deferred `copy_texture_to_texture` commands recorded by any
    // `grow()` calls above are still sitting unsubmitted in `encoder` at
    // this point. Submitting now is what lets them run - and, per the
    // hypothesis, run *after* every `write_texture` upload issued during
    // `prepare()`, potentially clobbering some of them.
    gpu.queue.submit(std::iter::once(encoder.finish()));

    let has_ink = render_and_read_ink(
        &gpu,
        &mut atlas,
        &renderer,
        &viewport,
        canvas_width,
        canvas_height,
    );

    let blank: Vec<&str> = cells
        .iter()
        .zip(has_ink.iter())
        .filter(|(_, ink)| !**ink)
        .map(|(cell, _)| cell.label.as_str())
        .collect();

    assert!(
        blank.is_empty(),
        "{} of {} glyphs rendered blank after a single-encoder burst \
         (grow()'s deferred atlas copy likely clobbered a `queue.write_texture` \
         upload that landed in the copied region before the copy executed). \
         Examples: {:?}",
        blank.len(),
        cells.len(),
        &blank[..blank.len().min(15)],
    );
}

/// Control: same glyphs, same fonts/sizes, but submitted in small batches
/// with an explicit `queue.submit` after each `prepare()` call. This still
/// forces the atlas to grow repeatedly, but every deferred copy from a given
/// batch's growth is executed (submitted) before the *next* batch's
/// `write_texture` calls happen, so no write should ever land in a region a
/// later-executing copy will still stomp over.
#[test]
fn batched_with_submits_between_does_not_corrupt_glyphs() {
    let gpu = Gpu::new();
    let mut font_system = FontSystem::new();
    let mut swash_cache = cryoglyph::SwashCache::new();
    let cache = Cache::new(&gpu.device);
    let mut viewport = cryoglyph::Viewport::new(&gpu.device, &cache);
    let mut atlas = TextAtlas::with_color_mode(
        &gpu.device,
        &gpu.queue,
        &cache,
        TextureFormat::Rgba8Unorm,
        ColorMode::Web,
    );
    let mut renderer = TextRenderer::new(&mut atlas, &gpu.device, MultisampleState::default(), None);

    let (buffers, cells) = build_buffers(&mut font_system);
    let canvas_width = COLS * CELL;
    let canvas_height = (cells.len() as u32).div_ceil(COLS) * CELL;
    viewport.update(
        &gpu.queue,
        Resolution {
            width: canvas_width,
            height: canvas_height,
        },
    );

    // `TextRenderer::prepare` clears and rebuilds the renderer's own vertex
    // buffer on every call, so - unlike the single-encoder burst test - we
    // must render and read back each batch's glyphs before moving on to the
    // next one (which will overwrite them in the renderer, though not in the
    // shared atlas).
    let mut blank: Vec<String> = Vec::new();

    const BATCH: usize = 4;
    for chunk_start in (0..buffers.len()).step_by(BATCH) {
        let chunk_end = (chunk_start + BATCH).min(buffers.len());
        let chunk_cells = &cells[chunk_start..chunk_end];
        let text_areas: Vec<TextArea> = buffers[chunk_start..chunk_end]
            .iter()
            .zip(chunk_cells.iter())
            .map(|(buffer, cell)| text_area(buffer, cell))
            .collect();

        let mut encoder = gpu
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("atlas-growth-race batch"),
            });

        renderer
            .prepare(
                &gpu.device,
                &gpu.queue,
                &mut encoder,
                &mut font_system,
                &mut atlas,
                &viewport,
                text_areas,
                &mut swash_cache,
            )
            .expect("prepare should not report AtlasFull for this glyph count");

        // Every deferred `copy_texture_to_texture` from a grow triggered by
        // *this* batch is submitted - and therefore executes - before the
        // next batch's `prepare()` issues any further `write_texture` calls.
        gpu.queue.submit(std::iter::once(encoder.finish()));

        let has_ink = render_and_read_ink(
            &gpu,
            &mut atlas,
            &renderer,
            &viewport,
            canvas_width,
            canvas_height,
        );

        for cell in chunk_cells {
            let index = (cell.row * (canvas_width / CELL) + cell.col) as usize;
            if !has_ink[index] {
                blank.push(cell.label.clone());
            }
        }
    }

    assert!(
        blank.is_empty(),
        "{} of {} glyphs rendered blank even with submits between batches: {:?}",
        blank.len(),
        cells.len(),
        &blank[..blank.len().min(15)],
    );
}
