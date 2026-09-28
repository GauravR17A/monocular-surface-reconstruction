'use client';

export type EvaluationEvidenceId = 'urban' | 'sparse' | 'hilly' | 'forest';

type LandscapeEvaluationProps = {
  open: boolean;
  onClose: () => void;
  onOpenEvidence: (evidence: EvaluationEvidenceId) => void;
};

type EvaluationRow = {
  category: string;
  scene: string;
  product: string;
  reference: string;
  rmse: string;
  mae: string;
  correlation: string;
  r2: string;
  status: 'VALIDATED' | 'FUNCTIONAL';
  note: string;
  evidence: EvaluationEvidenceId | null;
};

const ROWS: EvaluationRow[] = [
  {
    category: 'Urban',
    scene: 'US3D urban satellite tile',
    product: 'Metric nDSM',
    reference: 'Independent LiDAR-derived nDSM',
    rmse: '2.54 m',
    mae: '1.30 m',
    correlation: '0.746',
    r2: '0.507',
    status: 'VALIDATED',
    note: '262,144 aligned pixels · historical result; source imagery is restricted',
    evidence: null,
  },
  {
    category: 'Sparse',
    scene: 'Amsterdam rural fringe',
    product: 'Metric nDSM',
    reference: 'Held-out HighBuild height raster',
    rmse: '1.48 m',
    mae: '0.71 m',
    correlation: '0.873',
    r2: '0.752',
    status: 'VALIDATED',
    note: '1,048,576 aligned pixels · held-out test chip',
    evidence: 'sparse',
  },
  {
    category: 'Hilly',
    scene: 'Manali, Himachal Pradesh',
    product: 'Absolute DSM',
    reference: 'Independent SRTM 1 arc-second DEM',
    rmse: '15.03 m',
    mae: '10.24 m',
    correlation: '1.000',
    r2: '0.999',
    status: 'VALIDATED',
    note: '589,824 aligned pixels · 30 m coarse reference · 0.71% of scene relief',
    evidence: 'hilly',
  },
  {
    category: 'Forest',
    scene: 'Open-Canopy rural test tile',
    product: 'Canopy-height nDSM',
    reference: 'IGN LiDAR-derived canopy height',
    rmse: '4.22 m',
    mae: '2.86 m',
    correlation: '0.780',
    r2: '0.409',
    status: 'VALIDATED',
    note: '147,456 aligned pixels · official test split',
    evidence: 'forest',
  },
];

export function LandscapeEvaluation({
  open,
  onClose,
  onOpenEvidence,
}: LandscapeEvaluationProps) {
  if (!open) return null;
  return (
    <div className="evaluation-overlay" role="dialog" aria-modal="true" aria-label="Four-category evaluation table">
      <section className="evaluation-board">
        <header className="evaluation-header">
          <div>
            <p className="eyebrow">Reviewer evidence · landscape stability</p>
            <h2>Four-category evaluation matrix</h2>
            <p>Urban, sparse, hilly and forest coverage, with reference-backed metrics separated from functional demonstrations.</p>
          </div>
          <button onClick={onClose} type="button" aria-label="Close four-category evaluation">×</button>
        </header>

        <div className="evaluation-summary">
          <div><span>Demonstrated</span><strong>4 / 4</strong><small>required landscape categories</small></div>
          <div><span>Reference validated</span><strong>4 / 4</strong><small>independent height references</small></div>
          <div><span>Metric products</span><strong>nDSM + DSM</strong><small>relative and absolute workflows</small></div>
        </div>

        <div className="evaluation-table-wrap">
          <table>
            <thead>
              <tr><th>Landscape</th><th>Scene / product</th><th>Evaluation reference</th><th>RMSE</th><th>MAE</th><th>Corr.</th><th>R²</th><th>Status / evidence</th></tr>
            </thead>
            <tbody>
              {ROWS.map((row) => (
                <tr key={row.category}>
                  <th scope="row"><span className={`category-dot ${row.category.toLowerCase()}`} />{row.category}</th>
                  <td><strong>{row.scene}</strong><small>{row.product}</small></td>
                  <td><strong>{row.reference}</strong><small>{row.note}</small></td>
                  <td className="metric-cell">{row.rmse}</td>
                  <td className="metric-cell">{row.mae}</td>
                  <td className="metric-cell">{row.correlation}</td>
                  <td className="metric-cell">{row.r2}</td>
                  <td><span className={`evaluation-status ${row.status.toLowerCase()}`}>{row.status}</span><button disabled={row.evidence === null} onClick={() => row.evidence && onOpenEvidence(row.evidence)} type="button">{row.evidence ? 'View evidence' : 'Imagery restricted'}</button></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <footer className="evaluation-footer">
          <p><strong>Honesty rule:</strong> RMSE, MAE, correlation and R² are shown only when an independent reference exists and was excluded from inference.</p>
          <p>Manali uses Copernicus for calibration but separate SRTM for evaluation. Its 15.03 m RMSE is a coarse terrain check across 2,110 m of relief, not LiDAR validation of individual objects.</p>
        </footer>
      </section>
    </div>
  );
}
