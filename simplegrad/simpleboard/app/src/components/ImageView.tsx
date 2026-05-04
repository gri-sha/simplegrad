import { useMemo, useState, useEffect, useRef } from 'react';
import type { ImageInfo } from '../types';
import { colorForRun } from '../colors';
import { Image as ImageIcon } from 'lucide-react';

interface ImageViewProps {
  selectedRunIds: number[];
  runNames: Record<number, string>;
  imagesByRun: Record<number, Record<string, ImageInfo[]>>;
}

export function ImageView({ selectedRunIds, runNames, imagesByRun }: ImageViewProps) {
  const [selectedMetric, setSelectedMetric] = useState<string | null>(null);
  const [stepIndex, setStepIndex] = useState<number>(0);

  const availableMetrics = useMemo(() => {
    const keys = new Set<string>();
    for (const id of selectedRunIds) {
      const imgs = imagesByRun[id];
      if (imgs) Object.keys(imgs).forEach(k => keys.add(k));
    }
    return Array.from(keys).sort();
  }, [selectedRunIds, imagesByRun]);

  useEffect(() => {
    if (availableMetrics.length > 0 && (!selectedMetric || !availableMetrics.includes(selectedMetric))) {
      setSelectedMetric(availableMetrics[0]);
    }
  }, [availableMetrics, selectedMetric]);

  const allSteps = useMemo(() => {
    if (!selectedMetric) return [];
    const steps = new Set<number>();
    for (const id of selectedRunIds) {
      const imgs = imagesByRun[id]?.[selectedMetric] || [];
      imgs.forEach(h => steps.add(h.step));
    }
    return Array.from(steps).sort((a, b) => a - b);
  }, [selectedRunIds, imagesByRun, selectedMetric]);

  useEffect(() => {
    if (allSteps.length > 0) {
      setStepIndex(allSteps.length - 1); // default to latest
    }
  }, [allSteps.length]);

  const currentStep = allSteps[stepIndex];

  if (selectedRunIds.length === 0 || availableMetrics.length === 0) {
    return (
      <div className="main-empty">
        <ImageIcon size={48} strokeWidth={1} />
        <p>No images available for the selected run(s).</p>
      </div>
    );
  }

  return (
    <div className="hparams-container" style={{ padding: '1rem', width: '100%', maxWidth: '1000px', margin: '0 auto' }}>
      <div style={{ display: 'flex', gap: '1rem', marginBottom: '1rem', alignItems: 'center' }}>
        <select 
          value={selectedMetric || ''} 
          onChange={e => setSelectedMetric(e.target.value)}
          style={{ padding: '6px', borderRadius: '4px', border: '1px solid var(--border)' }}
        >
          {availableMetrics.map(m => <option key={m} value={m}>{m}</option>)}
        </select>
      </div>

      {allSteps.length > 0 && (
        <div style={{ marginBottom: '2rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px', fontSize: '0.875rem', color: 'var(--muted)' }}>
            <span>Step {allSteps[0]}</span>
            <span style={{ fontWeight: 600, color: 'var(--color-fg)' }}>Current Step: {currentStep}</span>
            <span>Step {allSteps[allSteps.length - 1]}</span>
          </div>
          <input 
            type="range" 
            min={0} 
            max={allSteps.length - 1} 
            value={stepIndex} 
            onChange={e => setStepIndex(Number(e.target.value))}
            style={{ width: '100%' }}
          />
        </div>
      )}

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem', justifyContent: 'center' }}>
        {selectedRunIds.map(id => {
          const imgs = imagesByRun[id]?.[selectedMetric!] || [];
          let closest = imgs[0];
          for (const img of imgs) {
            if (img.step <= currentStep) closest = img;
            else break;
          }
          if (!closest) return null;
          return (
            <div key={id} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '8px' }}>
              <span className="metric-graph-legend-item">
                <span className="metric-graph-legend-swatch" style={{ background: colorForRun(id) }} />
                {runNames[id]} (Step {closest.step})
              </span>
              <RawImage imgInfo={closest} />
            </div>
          );
        })}
      </div>
    </div>
  );
}

function RawImage({ imgInfo }: { imgInfo: ImageInfo }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const binStr = atob(imgInfo.data_b64);
    const len = binStr.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) bytes[i] = binStr.charCodeAt(i);

    const w = imgInfo.width;
    const h = imgInfo.height;
    const c = imgInfo.channels;
    
    canvas.width = w;
    canvas.height = h;

    const imgData = ctx.createImageData(w, h);
    const data = imgData.data;

    if (c === 1) {
      for (let i = 0; i < w * h; i++) {
        const val = bytes[i];
        data[i * 4] = val;
        data[i * 4 + 1] = val;
        data[i * 4 + 2] = val;
        data[i * 4 + 3] = 255;
      }
    } else if (c === 3) {
      for (let i = 0; i < w * h; i++) {
        data[i * 4] = bytes[i * 3];
        data[i * 4 + 1] = bytes[i * 3 + 1];
        data[i * 4 + 2] = bytes[i * 3 + 2];
        data[i * 4 + 3] = 255;
      }
    } else if (c === 4) {
      for (let i = 0; i < w * h * 4; i++) {
        data[i] = bytes[i];
      }
    }
    
    ctx.putImageData(imgData, 0, 0);
  }, [imgInfo]);

  return (
    <canvas 
      ref={canvasRef} 
      style={{ 
        border: '1px solid var(--border)', 
        borderRadius: '4px',
        maxWidth: '100%',
        maxHeight: '400px',
        objectFit: 'contain',
        imageRendering: 'pixelated'
      }} 
    />
  );
}
