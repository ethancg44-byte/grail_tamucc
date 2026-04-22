import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

runs = [
    ('01', 'LRASPP', 'CE', '-', '0.5/2.0/3.0', 50, 256, '-', '-', '-', '-'),
    ('02', 'LRASPP', 'CE', '-', '0.3/1.5/6.0', 50, 256, '-', '-', '-', '-'),
    ('03', 'LRASPP', 'CE', '-', '0.3/1.5/6.0', 50, 256, '-', '-', '-', '-'),
    ('04', 'LRASPP', 'CE', '-', '0.2/1.5/8.0', 50, 256, '-', '-', '-', '-'),
    ('05', 'LRASPP', 'CE', '-', '0.2/2.0/8.0', 50, 256, '-', '-', '-', '-'),
    ('06', 'LRASPP', 'CE', '-', '0.3/1.5/6.0', 100, 256, '-', '-', '-', '-'),
    ('07', 'LRASPP', 'CE', '-', '0.2/2.0/8.0', 100, 256, '-', '-', '-', '-'),
    ('08*', 'LRASPP', 'CE+Dice', '-', '0.5/2.0/3.0', 100, 256, '0.8865', '0.6750', '0.5541', '0.7135'),
    ('09', 'LRASPP', 'CE', '-', '0.5/2.0/3.0', 100, 256, '0.8817', '0.6793', '0.5260', '0.6957'),
    ('12', 'LRASPP', 'CE+Dice', '-', '0.5/2.0/3.0', 100, 256, '0.8866', '0.6764', '0.5437', '0.7022'),
    ('13', 'LRASPP', 'CE+Dice', '-', '0.5/2.0/3.0', 100, 256, '0.8914', '0.6808', '0.5482', '0.7068'),
    ('14', 'LRASPP', 'CE+Dice', 'strong', '0.5/2.0/3.0', 250, 256, '0.8854', '0.6717', '0.5310', '0.6960'),
    ('15', 'LRASPP', 'CE+Dice', 'strong', '0.5/2.0/3.0', 250, 256, '0.8833', '0.6694', '0.5267', '0.6931'),
    ('16', 'LRASPP', 'CE+Dice', 'strong', '0.5/2.0/3.0', 200, 256, '0.8750', '0.6614', '0.5118', '0.6827'),
    ('17', 'LRASPP', 'Lovasz', 'basic', '0.5/2.0/3.0', 200, 256, '0.7441', '0.0991', '0.0221', '0.2884'),
    ('18', 'DeepLabV3', 'CE+Dice', 'basic', '0.5/2.0/3.0', 200, 256, '0.8417', '0.6040', '0.4404', '0.6287'),
    ('19', 'DeepLabV3', 'CE+Lovasz', 'basic', '0.5/2.0/3.0', '71+', 256, '0.8577', '0.5723', '0.3846', '0.6049'),
    ('20', 'LRASPP', 'CE+Tversky', 'basic', '0.5/2.0/3.0', 100, 256, '0.8865', '0.6750', '0.5441', '0.7019'),
    ('21+', 'LRASPP', 'CE+Dice', 'basic', '0.5/2.0/3.0', 100, 256, '0.8953', '0.6903', '0.5559', '0.7139'),
    ('22', 'LRASPP', 'CE+Dice', 'basic', '0.5/2.0/3.0', 100, 256, '0.8927', '0.6926', '0.5513', '0.7122'),
    ('23', 'LRASPP', 'CE+Lovasz', 'basic', '0.5/2.0/3.0', 100, 256, '0.9070', '0.6935', '0.5010', '0.7005'),
    ('24', 'LRASPP', 'CE+Dice', 'basic', '0.5/2.0/3.0', 100, 320, '0.9099', '0.7235', '0.6066', '0.7467'),
    ('25', 'LRASPP', 'CE+Dice', 'basic', '0.5/2.0/3.0', 100, 384, '0.9224', '0.7554', '0.6533', '0.7770'),
    ('26', 'LRASPP', 'CE+Dice', 'basic', '0.5/2.0/3.0', 100, 256, '0.8913', '0.6858', '0.5490', '0.7087'),
    ('27', 'LRASPP', 'CE+Dice', 'basic', '0.5/2.0/3.0', 100, 256, '0.8906', '0.6851', '0.5435', '0.7064'),
    ('28', 'LRASPP', 'CE+Lovasz', 'basic', '0.5/2.0/3.0', 100, 256, '0.9047', '0.6907', '0.4932', '0.6962'),
    ('29', 'LRASPP', 'CE+Dice', 'basic', '0.5/2.0/3.0', 100, 448, '0.9295', '0.7718', '0.6781', '0.7931'),
    ('30', 'LRASPP', 'CE+Dice', 'basic', '0.5/2.0/3.0', 100, 512, '0.9380', '0.7906', '0.6810', '0.7950'),
    ('31', 'LRASPP', 'CE+Tversky', 'basic', '0.5/2.0/3.0', 150, 512, '0.9447', '0.8040', '0.7382', '0.8290'),
    ('32', 'LRASPP', 'CE+Tversky', 'basic', '0.5/2.0/3.0', 150, 640, '0.9505', '0.8224', '0.7652', '0.8460'),
    ('33', 'LRASPP', 'CE+Tversky', 'basic', '0.5/2.0/3.0', 150, 768, '0.9574', '0.8410', '0.7964', '0.8649'),
]

headers = ['Run', 'Model', 'Loss', 'Aug', 'Class Wts\n(BG/LF/ST)', 'Epochs', 'Res', 'BG IoU', 'Leaf IoU', 'Stem IoU', 'mIoU']
nrows = len(runs)
ncols = len(headers)
col_widths = [0.04, 0.08, 0.08, 0.05, 0.11, 0.05, 0.04, 0.07, 0.07, 0.07, 0.07]
total_table_w = sum(col_widths)  # 0.71

fig, ax = plt.subplots(figsize=(16, 13))
ax.axis('off')
fig.patch.set_facecolor('#0e1117')

# Tighter layout: table fills most of figure, small margins
fig.subplots_adjust(top=0.95, bottom=0.10, left=0.05, right=0.95)

cell_text = [[str(x) for x in r] for r in runs]

table = ax.table(
    cellText=cell_text,
    colLabels=headers,
    colWidths=col_widths,
    loc='upper center',
    cellLoc='center',
)
table.auto_set_font_size(False)
table.set_fontsize(8.5)
table.scale(1.0, 1.15)

for j in range(ncols):
    cell = table[0, j]
    cell.set_facecolor('#1e2433')
    cell.set_text_props(color='#c8d0e0', fontweight='bold', fontfamily='monospace', fontsize=9)
    cell.set_edgecolor('#3a3f55')
    cell.set_height(cell.get_height() * 1.8)  # taller header row for two-line labels

for i in range(nrows):
    run_id = runs[i][0]
    is_best = run_id == '33'
    is_good = run_id in ('31', '32')
    if is_best:
        bg, txt, fw = '#1a2e1a', '#55ff55', 'bold'
    elif is_good:
        bg, txt, fw = '#162035', '#77bbee', 'normal'
    elif i % 2 == 0:
        bg, txt, fw = '#13161f', '#b0b8c8', 'normal'
    else:
        bg, txt, fw = '#181c28', '#b0b8c8', 'normal'
    for j in range(ncols):
        cell = table[i + 1, j]
        cell.set_facecolor(bg)
        cell.set_edgecolor('#2a2f40')
        cell.set_text_props(color=txt, fontfamily='monospace', fontsize=8.5, fontweight=fw)

# Get table bounding box to position footer directly below
fig.canvas.draw()
bbox = table.get_window_extent(fig.canvas.get_renderer())
bbox_fig = bbox.transformed(fig.transFigure.inverted())
table_left = bbox_fig.x0
table_bottom = bbox_fig.y0

ax.set_title('GRAIL Plant Segmentation - Experiment Results (Runs 01-33)',
             fontsize=14, fontweight='bold', fontfamily='monospace', color='#d0d8e8', pad=12)

# Place footer just below the table, aligned to table left edge
footer_lines = [
    '* Best run (baseline)   + OOM crash   + Still training (stem_sampling enabled)',
    'Target: Stem IoU >= 0.70  |  Dataset: 8000 train / 1000 val / 1000 test  |  3 classes: BG/LEAF/STEM',
    'Runs 01-07: No CSV logs (pre-logging). All used CE loss with various weight sweeps - all worse than run 08.',
    'Runs 26-27: LR and warmup ablations at 256px. Run 33 is the overall best (mIoU=0.8649).',
]
footer_text = '\n'.join(footer_lines)
fig.text(table_left, table_bottom - 0.01, footer_text,
         fontsize=7.5, fontfamily='monospace', color='#778899', va='top')

out_path = 'checkpoints/run33_cetversky_stemsamp_768/simulation_results_updated.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor(), pad_inches=0.15)
plt.close()

# Auto-crop
img = Image.open(out_path)
arr = np.array(img)
mask = ~((arr[:,:,0] < 25) & (arr[:,:,1] < 25) & (arr[:,:,2] < 35))
rows_mask = np.any(mask, axis=1)
cols_mask = np.any(mask, axis=0)
rmin, rmax = np.where(rows_mask)[0][[0, -1]]
cmin, cmax = np.where(cols_mask)[0][[0, -1]]
pad = 20
cropped = img.crop((max(0,cmin-pad), max(0,rmin-pad), min(arr.shape[1],cmax+pad), min(arr.shape[0],rmax+pad)))
cropped.save(out_path)
print(f'Done: {cropped.size}')
