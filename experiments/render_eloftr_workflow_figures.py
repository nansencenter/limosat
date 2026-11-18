#!/usr/bin/env python3
"""Render minimal real-data figures from a bounded current-ELoFTR rerun.
"""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import importlib
import json
from pathlib import Path
import sys
from types import ModuleType

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.tri import Triangulation
from matplotlib.patches import Circle, ConnectionPatch, Rectangle
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial import Delaunay, cKDTree

BG, PANEL, INK, MUTED = '#f6f8fa', '#eef2f5', '#142d3c', '#516570'
TEAL, CYAN, WARM, GREY, RED = '#005a70', '#42ffe6', '#b95713', '#aab5bc', '#cc3e70'
IDS = [6775, 6801, 6901, 6998]
LABEL = dict(zip(IDS, 'ABCD'))
BOUNDS = (768000., 864000., -680000., -584000.)
RECORDS, INPUTS = [], set()
PDF_EXPORT, SVG_EXPORT = None, True
HALO = [pe.Stroke(linewidth=3.2, foreground='#14212b'), pe.Normal()]


def read_json(path):
    INPUTS.add(path)
    return json.loads(path.read_text())


def load_core(checkout):
    # Skip GPU package initialization; import unchanged scientific CPU modules.
    package = ModuleType('workflow_core')
    package.__path__ = [str(checkout / 'limosat')]
    sys.modules[package.__name__] = package
    return {name: importlib.import_module(f'workflow_core.{name}')
            for name in ('config', 'catalog', 'models', 'field', 'imagery', 'deformation', 'routing')}


def page(number, title, subtitle, panels=2):
    titles={1:'Image pair',2:'Matches',3:'Matches → field',4:'Local consensus',
            5:'Pair contributions',6:'Advection',7:'Deformation',8:'Deformation',
            '01a':'Tiles'}
    fig = plt.figure(figsize=(14.4, 8.1), facecolor=BG)
    label=f'{number:02d}' if isinstance(number,int) else number
    fig.text(.045, .951, label, color=MUTED, size=12)
    fig.text(.09, .945, titles.get(number,title), color=INK, size=21)
    width = .87 / panels
    axes = [fig.add_axes([.065 + i * (width + .015), .13, width-.025, .71], facecolor=BG)
            for i in range(panels)]
    return fig, axes


def finish(fig, out, name, headline, detail, caption):
    fig.savefig(out / f'{name}.png', dpi=190, facecolor=BG)
    if SVG_EXPORT:fig.savefig(out / f'{name}.svg', facecolor=BG)
    if PDF_EXPORT is not None:PDF_EXPORT.savefig(fig, facecolor=BG, dpi=190)
    RECORDS.append(dict(name=name, caption=caption, notes=[headline,detail]))
    plt.close(fig)
    print(name, flush=True)


def alignment_flow(fig, active, phase_pixel_size):
    labels=[f'Phase correlation · {phase_pixel_size:g} m/px',
            'ELoFTR alignment · 320 m/px', 'ELoFTR matching · 80 m/px']
    for index,(x,label) in enumerate(zip((.20,.50,.80),labels)):
        fig.text(x,.885,label,ha='center',size=11,
                 color=TEAL if index==active else MUTED,
                 weight='bold' if index==active else 'normal')
    for x0,x1 in ((.335,.365),(.635,.665)):
        fig.add_artist(ConnectionPatch((x0,.89),(x1,.89),'figure fraction',
            'figure fraction',arrowstyle='-|>',mutation_scale=12,color=INK,lw=1.2))


def center_shift_axis(ax, start, end, start_label, end_label, bounds):
    ax.set(xlim=bounds[:2],ylim=bounds[2:],aspect='equal',
           xlabel='East offset (km)',ylabel='North offset (km)')
    ax.grid(color='#d4dee4',lw=.7)
    ax.scatter(*start,s=80,c=TEAL,ec='white',lw=1.2,zorder=4)
    ax.scatter(*end,s=80,c=WARM,ec='white',lw=1.2,zorder=4)
    ax.annotate('',xy=end,xytext=start,
                arrowprops=dict(arrowstyle='-|>',color=INK,lw=2,mutation_scale=17))
    for point,label,offset in ((start,start_label,(10,8)),(end,end_label,(10,-18))):
        ax.annotate(label,point,xytext=offset,textcoords='offset points',size=10,color=INK)
    ax.set_title('Tile-centre offset',size=11,pad=12,color=INK)


def focused_outputs(out, run, data, args, selections, checks):
    global PDF_EXPORT
    if PDF_EXPORT is not None:
        PDF_EXPORT.close();PDF_EXPORT=None
    np.savez_compressed(out/'display_selections.npz',**selections)
    manifest=dict(dataset='N-ICE2015; current ELoFTR real-data rerun',
        scope='Steps 1–4 only',data=str(data.resolve()),source_commit=run['source_commit'],
        crs='EPSG:3413',checks=checks,figures=RECORDS,
        pdf=str(args.pdf.resolve()) if args.pdf else None,
        reproduction={'script':str(Path(__file__).resolve()),
          'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'arguments':sys.argv[1:]})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    overview=Image.new('RGB',(1800,338*int(np.ceil(len(RECORDS)/3))),BG)
    for i,r in enumerate(RECORDS):
        im=Image.open(out/(r['name']+'.png')).convert('RGB');im.thumbnail((600,338))
        overview.paste(im,((i%3)*600,(i//3)*338))
    overview.save(out/'00_steps_1_to_4_overview.png')
    print(json.dumps(checks,indent=2))


def legacy_alignment_figure(core, paths, center, initial, refined, coarse, fine, out, done):
    prior_fine,_=core['imagery'].north_up_patch(paths[6801],tuple(center+initial),512,80.)
    corrected,_=core['imagery'].north_up_patch(paths[6801],tuple(center+refined),512,80.)
    yy,xx=np.indices((512,512));checker=(xx//64+yy//64)%2==0
    fig,axes=page('01c','Initial alignment','',panels=3)
    axes[0].imshow(coarse['source'],cmap='gray',vmin=0,vmax=255)
    axes[0].add_patch(Rectangle((191.5,191.5),128,128,fill=False,ec=CYAN,lw=2,path_effects=HALO))
    cp=coarse['raw_source_px'];sample=np.linspace(0,len(cp)-1,min(140,len(cp)),dtype=int)
    axes[0].scatter(cp[sample,0],cp[sample,1],s=15,c=CYAN,ec='#101820',lw=.8)
    axes[0].set_title('Coarse context · 320 m/px',size=11,pad=12)
    for ax,target,title in zip(axes[1:],(prior_fine,corrected),('Before · 80 m/px','After · 80 m/px')):
        ax.imshow(np.where(checker,fine['source'],target),cmap='gray',vmin=0,vmax=255)
        ax.set_title(title,size=11,color=INK,pad=12)
    for ax in axes:ax.set_xticks([]);ax.set_yticks([])
    done(fig,'01c_coarse_alignment',f'Actual coarse context and fine-scale checkerboards before/after the recorded refinement. Initial offset {initial.tolist()} m; refined offset {refined.tolist()} m.')


def inside(xy, bounds=BOUNDS, margin=0):
    x0,x1,y0,y1 = bounds
    return ((xy[:,0]>=x0+margin)&(xy[:,0]<=x1-margin)&
            (xy[:,1]>=y0+margin)&(xy[:,1]<=y1-margin))


def map_axis(ax, title, bounds=BOUNDS):
    ax.set(xlim=np.array(bounds[:2])/1000, ylim=np.array(bounds[2:])/1000,
           aspect='equal', title=title, xlabel='Easting (km)', ylabel='Northing (km)')
    ax.set_title(title, color=INK, size=11, pad=11)
    ax.tick_params(labelsize=8.5, colors=MUTED)
    ax.xaxis.label.set(size=9, color=MUTED)
    ax.yaxis.label.set(size=9, color=MUTED)
    for spine in ax.spines.values(): spine.set_color('#b6c3ca')


def dots(ax, xy, color=TEAL, size=35, marker='o', zorder=6):
    # White outer rim, charcoal inner outline, and coloured centre: readable
    # on both dark leads and bright ice at presentation scale.
    if not len(xy): return
    ax.scatter(xy[:,0]/1000,xy[:,1]/1000,c='white',s=size*1.65,marker=marker,linewidths=0,zorder=zorder)
    ax.scatter(xy[:,0]/1000,xy[:,1]/1000,color=color,s=size,marker=marker,
               edgecolors='#101820',linewidths=.8,zorder=zorder+.1)


def arrows(ax, xy, d, color=TEAL, alpha=1., width=.004):
    if not len(xy): return
    artist=ax.quiver(xy[:,0]/1000,xy[:,1]/1000,d[:,0]/1000,d[:,1]/1000,
        angles='xy',scale_units='xy',scale=1,color=color,width=width,
        headwidth=3.6,headlength=4.5,zorder=5,alpha=alpha)
    artist.set_path_effects([pe.Stroke(linewidth=.7,foreground='white'),pe.Normal()])


def balanced(xy, bounds, maximum=90):
    """Deterministic representative per spatial cell; preserve array indexes."""
    candidates=np.flatnonzero(inside(xy,bounds,margin=1000))
    if len(candidates)<=maximum:return candidates
    n=max(2,int(np.sqrt(maximum)))
    lo=np.array([bounds[0],bounds[2]])
    span=np.array([bounds[1]-bounds[0],bounds[3]-bounds[2]])
    cells=np.clip(((xy[candidates]-lo)/span*n).astype(int),0,n-1)
    keys=cells[:,1]*n+cells[:,0]
    centres=lo+(cells+.5)*span/n
    dist=np.linalg.norm(xy[candidates]-centres,axis=1)
    order=np.lexsort((candidates,dist,keys))
    _,first=np.unique(keys[order],return_index=True)
    return candidates[order[first]]


def triangles(field):
    good=field.available
    xy=field.source_xy_m[good]
    target=xy+field.displacement_m[good]
    tri=Delaunay(xy).simplices
    verts=xy[tri]; dest=target[tri]
    edge=np.linalg.norm(verts-np.roll(verts,1,axis=1),axis=2).max(1)
    def area(t):
        a=t[:,1]-t[:,0];b=t[:,2]-t[:,0]
        return a[:,0]*b[:,1]-a[:,1]*b[:,0]
    keep=(edge<=6400)&(area(verts)*area(dest)>0)
    return verts[keep]


def draw_mesh(ax,field,bounds=BOUNDS):
    verts=triangles(field)
    view=inside(verts.reshape(-1,2),bounds).reshape(-1,3).any(1)
    v=verts[view]/1000
    segments=np.concatenate([v[:,[0,1]],v[:,[1,2]],v[:,[2,0]]])
    artist=LineCollection(segments,colors='#c4cfd5',lw=.65,alpha=.7,zorder=3)
    ax.add_collection(artist)


def main():
    global PDF_EXPORT, SVG_EXPORT
    ap=argparse.ArgumentParser(description='Render current ELoFTR real-data workflow figures.')
    ap.add_argument('--demo-dir',type=Path,required=True)
    ap.add_argument('--source-checkout',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--first-four',action='store_true',help='Only render steps 1–4, with explicit alignment stages.')
    ap.add_argument('--pdf',type=Path,help='Combined scientific-figure PDF.')
    args=ap.parse_args(); data=args.demo_dir; out=args.output_dir
    out.mkdir(parents=True,exist_ok=True)
    SVG_EXPORT=not args.first_four
    if args.pdf:
        args.pdf.parent.mkdir(parents=True,exist_ok=True)
        PDF_EXPORT=PdfPages(args.pdf,metadata={'Title':'LiMOSAT ELoFTR — steps 1–4',
            'Author':'LiMOSAT','Subject':'Real-data tile alignment, matching and consensus'})
    core=load_core(args.source_checkout)
    run=read_json(data/'run_manifest.json')
    assert run['complete']
    for name,digest in run['source_hashes'].items():
        assert hashlib.sha256((args.source_checkout/'limosat'/name).read_bytes()).hexdigest()==digest
    plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','axes.unicode_minus':True})
    images={}; paths={}
    for key in IDS:
        row=run['imagery'][str(key)]; paths[key]=Path(row['path'])
        images[key]=core['catalog'].ImageRecord(str(key),paths[key],datetime.fromisoformat(row['time']))
    fields={}; matches={}
    for pair_id in run['pairs']:
        a,b=map(int,pair_id.split('__')); pair=core['catalog'].ImagePair(images[a],images[b])
        fields[a,b]=core['models'].DisplacementField(pair_id,str(a),str(b),pair.source.time_utc,
            pair.target.time_utc,**dict(np.load(data/pair_id/'field.npz')))
        matches[a,b]=core['models'].MotionMatches(**dict(np.load(data/pair_id/'matches.npz')))
        assert fields[a,b].checksum==run['pairs'][pair_id]['field_checksum']
    first=fields[6775,6801]; m=matches[6775,6801]; src=m.source_xy_m
    points=pd.DataFrame(read_json(data/'trajectories.json'))
    extensions=pd.DataFrame(read_json(data/'extensions.json'))
    cfg=core['config'].FieldConfig(**run['config']['field'])
    cache={}
    def background(ax,key,bounds=BOUNDS):
        cache_key=(key,tuple(bounds))
        if cache_key not in cache:
            n=int(round((bounds[1]-bounds[0])/80))
            patch,valid=core['imagery'].north_up_patch(paths[key],
                ((bounds[0]+bounds[1])/2,(bounds[2]+bounds[3])/2),n,80.)
            lo,hi=np.percentile(patch[valid],[2,98])
            cache[cache_key]=(np.ma.array(patch,mask=~valid),lo,hi)
        patch,lo,hi=cache[cache_key]
        ax.imshow(patch,cmap='gray',vmin=lo,vmax=hi,extent=np.array(bounds)/1000,
                  origin='upper',interpolation='nearest',zorder=1)
    def date(key):return f'{LABEL[key]} · {images[key].time_utc:%d %b %H:%M}'
    def done(fig,name,caption):finish(fig,out,name,'','',caption)
    detail=(804000.,828000.,-640000.,-616000.)
    fig,axes=page(1,'Image pair','')
    for ax,key in zip(axes,IDS[:2]):
        map_axis(ax,date(key)); background(ax,key)
        ax.add_patch(Rectangle((detail[0]/1000,detail[2]/1000),24,24,fill=False,
                              ec=CYAN,lw=2,path_effects=HALO,zorder=5))
    done(fig,'01_overlapping_images','Actual 28–29 April 2015 Sentinel-1 VAE inputs in EPSG:3413. Rectangle marks the detail. Display contrast only is stretched; matcher inputs are unchanged.')

    # Actual globally anchored source-tile cores and the selected 512 px tile.
    tiles=data/'6775__6801'/'tiles'
    alignment=dict(np.load(data/'6775__6801'/'alignment_0.npz'))
    chosen=int(np.argmin(np.linalg.norm(alignment['centers_xy_m']-[816000,-624000],axis=1)))
    tile_id=int(alignment['tile_ids'][chosen]); center=alignment['centers_xy_m'][chosen]
    coarse=dict(np.load(tiles/f'coarse_{tile_id:02d}_0.npz'))
    fine=dict(np.load(tiles/f'fine_{tile_id:02d}_0.npz'))
    fig,(ax,bx)=page('01a' if args.first_four else '01b','Tiles','')
    map_axis(ax,'Tile division');background(ax,6775)
    for xy in alignment['centers_xy_m']:
        r=Rectangle((xy[0]/1000-17.92,xy[1]/1000-17.92),35.84,35.84,
                    fill=False,ec='white',lw=1.15,path_effects=HALO,zorder=4)
        ax.add_patch(r)
    ax.add_patch(Rectangle((center[0]/1000-20.48,center[1]/1000-20.48),40.96,40.96,
                          fill=False,ec=CYAN,lw=2.2,path_effects=HALO,zorder=5))
    bx.imshow(fine['source'],cmap='gray',vmin=0,vmax=255)
    bx.add_patch(Rectangle((31.5,31.5),448,448,fill=False,ec=CYAN,lw=2.5,path_effects=HALO))
    bx.set(xlim=(-.5,511.5),ylim=(511.5,-.5),xlabel='Pixels',ylabel='Pixels',title='512 × 512 px · 80 m/px')
    bx.set_xticks([0,32,256,480,512]);bx.set_yticks([0,32,256,480,512])
    bx.text(256,256,'448 × 448',ha='center',va='center',color=CYAN,size=14,path_effects=HALO)
    bx.text(256,16,'32 px margin',ha='center',va='center',color='white',size=10,path_effects=HALO)
    done(fig,'01a_tile_division' if args.first_four else '01b_tile_division','The current processor uses 35.84 km source cores with a 2.56 km sampling margin on each side. Tile layout and sampled raster are from the new run; source ownership uses the central 448×448 pixels.')

    # Coarse context, followed by a fine-scale view of the recorded alignment.
    initial=alignment['initial_shift_m'][chosen]; refined=alignment['refined_shift_m'][chosen]
    phase_pixel_size=None
    if args.first_four:
        # Reconstruct the actual broad-context phase-correlation inputs and
        # verify the result against the recorded tile prior.
        import math
        import cv2
        overlap=core['imagery'].projected_footprint(paths[6775]).intersection(
            core['imagery'].projected_footprint(paths[6801])).buffer(0)
        minx,miny,maxx,maxy=overlap.bounds
        elapsed=(images[6801].time_utc-images[6775].time_utc).total_seconds()
        maximum=run['config']['matcher']['maximum_speed_m_per_day']*elapsed/86400
        extent=max(maxx-minx,maxy-miny)+2*maximum
        phase_pixel_size=max(1000.,extent/1024)
        pixels=max(32,min(1024,int(math.ceil(extent/phase_pixel_size/32)*32)))
        phase_center=((minx+maxx)/2,(miny+maxy)/2)
        phase_source,sv=core['imagery'].north_up_patch(paths[6775],phase_center,pixels,phase_pixel_size)
        phase_target,tv=core['imagery'].north_up_patch(paths[6801],phase_center,pixels,phase_pixel_size)
        mask=sv&tv;normalized=[]
        for patch in (phase_source,phase_target):
            values=patch[mask].astype(np.float32);values_out=np.zeros_like(patch,dtype=np.float32)
            values_out[mask]=(values-values.mean())/max(float(values.std()),1e-6)
            normalized.append(values_out)
        shift,response=cv2.phaseCorrelate(normalized[0],normalized[1],
            cv2.createHanningWindow((pixels,pixels),cv2.CV_32F))
        phase_reproduced=np.array([shift[0],-shift[1]])*phase_pixel_size
        if np.linalg.norm(phase_reproduced)>maximum:phase_reproduced*=maximum/np.linalg.norm(phase_reproduced)
        np.testing.assert_allclose(phase_reproduced,initial,atol=1e-6)
        cm=core['models'].MotionMatches(coarse['source_xy_m'],coarse['target_xy_m'],
            coarse['score'],np.full(len(coarse['score']),tile_id),np.full(len(coarse['score']),tile_id))
        refined_reproduced,reason=core['routing'].coarse_match_shift(cm,center,
            run['config']['routing']['coarse_support_radius_m'],
            run['config']['routing']['coarse_minimum_matches'],maximum)
        assert refined_reproduced is not None,reason
        np.testing.assert_allclose(refined_reproduced,refined,atol=1e-6)
        phase_corrected,_=core['imagery'].north_up_patch(paths[6801],
            tuple(np.asarray(phase_center)+initial),pixels,phase_pixel_size)
        yy,xx=np.indices(phase_source.shape);checker=(xx//64+yy//64)%2==0
        fig,axes=page('01b','Phase correlation','',panels=3)
        axes[2].set_position([.705,.13,.235,.71])
        alignment_flow(fig,0,phase_pixel_size)
        for ax,target,title in zip(axes[:2],(phase_target,phase_corrected),('Before','After')):
            ax.imshow(np.where(checker,phase_source,target),cmap='gray',vmin=0,vmax=255)
            ax.set_title(title,size=11,color=INK,pad=12);ax.set_xticks([]);ax.set_yticks([])
        center_shift_axis(axes[2],np.zeros(2),initial/1000,'A centre','B centre',(-5,5,-10,2))
        done(fig,'01b_phase_correlation',f'Actual broad-context phase-correlation inputs, {pixels}×{pixels} pixels at {phase_pixel_size:g} m/px, before/after routing. Checkerboards alternate A/B. The source-image and target-image sampling centres differ by the recorded phase offset {initial.tolist()} m. Phase correlation is on the image-overlap context and initializes all tile centres; it is not the 320 m/px ELoFTR pass. Reproduced response {response}.')
        coarse_corrected,_=core['imagery'].north_up_patch(paths[6801],tuple(center+refined),512,320.)
        yy,xx=np.indices((512,512));checker=(xx//64+yy//64)%2==0
        fig,axes=page('01c','ELoFTR alignment','',panels=3)
        axes[2].set_position([.705,.13,.235,.71])
        alignment_flow(fig,1,phase_pixel_size)
        for ax,target,title in zip(axes[:2],(coarse['target'],coarse_corrected),('Phase prior','ELoFTR refinement')):
            ax.imshow(np.where(checker,coarse['source'],target),cmap='gray',vmin=0,vmax=255)
            ax.set_title(title,size=11,color=INK,pad=12);ax.set_xticks([]);ax.set_yticks([])
            ax.add_patch(Rectangle((191.5,191.5),128,128,fill=False,ec=CYAN,lw=2,path_effects=HALO))
        center_shift_axis(axes[2],initial/1000,refined/1000,'Phase prior','Refined',(-3.5,3.5,-10.5,-3.5))
        done(fig,'01c_eloftr_320m_alignment',f'Actual 512×512 coarse source/target tiles at 320 m/px (163.84 km context). Left uses the phase prior; middle uses the recorded 320 m/px ELoFTR refinement. Cyan rectangle is the 40.96 km detailed window. Right arrow is the true refinement of target sampling centre, {(refined-initial).tolist()} m, before detailed 80 m/px matching.')
    else:
        legacy_alignment_figure(core,paths,center,initial,refined,coarse,fine,out,done)

    # Truly unfiltered matcher outputs, with selection solely for display.
    p=core['imagery'].projected_coordinates(fine['raw_source_px'],tuple(center),512,80.)
    q=core['imagery'].projected_coordinates(fine['raw_target_px'],tuple(center+fine['shift_m']),512,80.)
    visible=inside(p,detail,margin=1000)&inside(q,detail,margin=1000)
    selected=np.flatnonzero(visible)[balanced(p[visible],detail,maximum=36)]
    assert len(selected)>=8
    fig,axes=page(2,'Matches','')
    if args.first_four:alignment_flow(fig,2,phase_pixel_size)
    for ax,key,xy in zip(axes,IDS[:2],(p[selected],q[selected])):
        map_axis(ax,date(key),detail);background(ax,key,detail);dots(ax,xy,color=CYAN,size=55)
    if args.first_four:axes[1].set_ylabel('')
    marked=selected[np.linspace(0,len(selected)-1,6,dtype=int)]
    colors=['#ffdc52','#f47bb8','#7eff63','#5de6ff','#ff994e','#c9a1ff']
    for number,(i,color) in enumerate(zip(marked,colors),1):
        for ax,xy in zip(axes,(p[i],q[i])):
            dots(ax,xy[None],color=color,size=95,zorder=8)
            t=ax.annotate(str(number),xy/1000,xytext=(7,7),textcoords='offset points',color='white',size=11,zorder=10)
            t.set_path_effects([pe.Stroke(linewidth=3,foreground='#101820'),pe.Normal()])
        fig.add_artist(ConnectionPatch(p[i]/1000,q[i]/1000,'data','data',axesA=axes[0],axesB=axes[1],
            color=color,lw=1.25,zorder=7,path_effects=[pe.Stroke(linewidth=2.7,foreground='#101820'),pe.Normal()]))
    done(fig,'02_actual_point_matches',f'{len(selected)} spatially balanced rows from the selected tile’s genuinely unfiltered fine ELoFTR output. Six numbered correspondences identify the same rows on both images; geometry is unchanged. All raw rows are retained in data/tiles.')

    fig,(ax,bx)=page(3,'Matches → field','')
    map_axis(ax,'Point matches · A → B');map_axis(bx,'4 km field · A → B')
    raw=inside(src);dense=src[raw]
    ax.scatter(dense[:,0]/1000,dense[:,1]/1000,s=5,c=TEAL,alpha=.85,lw=0,zorder=3,rasterized=True)
    selected_vectors=balanced(src,BOUNDS,maximum=100)
    arrows(ax,src[selected_vectors],m.displacement_m[selected_vectors],color=WARM,width=.0038)
    dots(ax,src[selected_vectors],color=WARM,size=15)
    view=inside(first.source_xy_m); valid=view&first.available
    draw_mesh(bx,first)
    arrows(bx,first.source_xy_m[valid],first.displacement_m[valid],color=TEAL,width=.0031)
    dots(bx,first.source_xy_m[valid],color=TEAL,size=17)
    bad=first.source_xy_m[view&~first.available]
    bx.scatter(bad[:,0]/1000,bad[:,1]/1000,c=GREY,marker='x',s=20,lw=.9,zorder=4)
    done(fig,'03_matches_to_grid',f'{raw.sum():,} pre-consensus matches in view, after ownership, valid-endpoint and speed gates; {valid.sum()} supported 4 km nodes. Every in-view match is plotted on a plain background. Only arrow density is reduced. Grey crosses are unavailable nodes. Displacement arrows use true lengths.')

    # Find a stored node with an independently reproduced local estimate.
    tree=cKDTree(src); choice=None
    for node in sorted(np.flatnonzero(valid),key=lambda i:np.linalg.norm(first.source_xy_m[i]-[816000,-624000])):
        query=first.source_xy_m[node]
        distances,neighbours=tree.query(query,k=12,distance_upper_bound=6000.)
        neighbours=neighbours[np.isfinite(distances)]
        if len(neighbours)!=12:continue
        vectors=m.displacement_m[neighbours];weights=np.maximum(m.score[neighbours],1e-12)
        sep=np.linalg.norm(vectors[:,None]-vectors[None,:],axis=2)
        pivot=np.argmax((sep<=1000)@weights);agree=sep[pivot]<=1000
        estimated=core['field'].estimate_queries(m,query[None],cfg)
        if estimated['available'][0] and np.linalg.norm(estimated['displacement_m'][0]-first.displacement_m[node])<1e-5:
            choice=(node,neighbours,agree,pivot,estimated)
            if agree.sum()<12:break
    assert choice is not None
    node,neighbours,agree,pivot,estimated=choice;query=first.source_xy_m[node]
    local=src[neighbours];vectors=m.displacement_m[neighbours];est=estimated['displacement_m'][0]
    bounds=(query[0]-7000,query[0]+7000,query[1]-7000,query[1]+7000)
    fig,(ax,bx)=page(4,'Local consensus','')
    map_axis(ax,'12 neighbours',bounds)
    dots(ax,local[agree],size=55);dots(ax,local[~agree],color=RED,size=55)
    dots(ax,query[None],color=WARM,size=95,marker='s')
    ax.add_patch(Circle(query/1000,6,fill=False,ec=GREY,lw=1.2,ls='--'))
    bx.set_aspect('equal');v=vectors/1000
    bx.scatter(v[agree,0],v[agree,1],c=TEAL,s=65,ec='white',lw=1)
    bx.scatter(v[~agree,0],v[~agree,1],c=RED,s=75,marker='x',lw=1.6)
    bx.add_patch(Circle(v[pivot],1,fill=False,ec=TEAL,lw=1.4,ls='--'))
    bx.scatter(*(est/1000),c=WARM,s=110,marker='s',ec=INK,lw=.7,zorder=5)
    lo=np.minimum(v.min(0),v[pivot]-1.2);hi=np.maximum(v.max(0),v[pivot]+1.2)
    mid=(lo+hi)/2;span=max(hi-lo)*1.2
    bx.set(xlim=(mid[0]-span/2,mid[0]+span/2),ylim=(mid[1]-span/2,mid[1]+span/2),
           xlabel='Eastward displacement (km)',ylabel='Northward displacement (km)',title=f'{agree.sum()} / 12 agree')
    bx.grid(color='#d4dee4',lw=.7)
    consensus_error=float(np.linalg.norm(est-first.displacement_m[node]))
    done(fig,'04_actual_local_consensus','The square is the query and its estimate; teal is the selected agreement group, pink the other neighbours. Search radius 6 km, agreement distance 1 km, minimum 8 of 12. The stored weighted geometric median is independently reproduced.')
    if args.first_four:
        focused_outputs(out,run,data,args,
            {'raw_match_rows':selected_vectors,'raw_detail_rows':marked,
             'consensus_node':node,'consensus_neighbours':neighbours,'consensus_agree':agree,
             'phase_shift_m':initial,'coarse_eloftr_shift_m':refined},
            {'field_checksums_verified':len(fields),'consensus_error_m':consensus_error,
             'phase_reproduction_error_m':float(np.linalg.norm(phase_reproduced-initial)),
             'coarse_eloftr_reproduction_error_m':float(np.linalg.norm(refined_reproduced-refined)),
             'phase_input_pixel_size_m':phase_pixel_size,'phase_input_pixels':pixels,
             'alignment_pixel_size_m':320,'detailed_matching_pixel_size_m':80})
        return

    # Contributions are selected by the current trajectory composer, not averaged.
    target=points[(points.image_id=='6901')&points.x_m.notna()]
    fig,axes=page(5,'Pair contributions','',panels=3)
    contribution_counts={}
    for ax,key,color in zip(axes[:2],((6801,6901),(6775,6901)),(TEAL,WARM)):
        f=fields[key];available=f.available&inside(f.source_xy_m+np.nan_to_num(f.displacement_m))
        endpoints=f.source_xy_m[available]+f.displacement_m[available]
        map_axis(ax,f'{LABEL[key[0]]} → C');dots(ax,endpoints,color=color,size=17)
    ax=axes[2];map_axis(ax,'Selected at C')
    for key,color in (((6801,6901),TEAL),((6775,6901),WARM)):
        rows=target[target.source_pair_id==f'{key[0]}__{key[1]}']
        xy=rows[['x_m','y_m']].to_numpy(float);xy=xy[inside(xy)]
        dots(ax,xy,color=color,size=23);contribution_counts[f'{LABEL[key[0]]}→C']=len(xy)
    ax.scatter([],[],color=TEAL,label='B → C');ax.scatter([],[],color=WARM,label='A → C')
    ax.legend(loc='upper right',frameon=False,fontsize=10)
    done(fig,'05_other_pair_contributions','Incoming pair fields are independent. Third panel uses the actual current trajectory composer’s selected measured links, including missing-node fallback. Teal: B→C; orange: A→C. New seeds at C are omitted. No pair fields are averaged.')

    # Follow an initially regular set of real material-point identities.
    seeds=points[(points.image_id=='6775')&(points.state=='created')].copy()
    seed_xy=seeds[['x_m','y_m']].to_numpy(float)
    focus=(804000.,828000.,-636000.,-612000.)
    seeds=seeds[inside(seed_xy,focus)]
    seed_ids=set(seeds.trajectory_id)
    initial_xy=seeds.set_index('trajectory_id')[['x_m','y_m']]
    mesh_edges=[]
    seed_order=list(initial_xy.index);sx=initial_xy.to_numpy(float)
    for i in range(len(sx)):
        for j in range(i+1,len(sx)):
            if np.isclose(np.linalg.norm(sx[i]-sx[j]),4000,atol=.01):mesh_edges.append((seed_order[i],seed_order[j]))
    fig,axes=page(6,'Advection','',panels=3)
    moving=points[points.image_id.isin(['6775','6901','6998'])&points.trajectory_id.isin(seed_ids)&points.x_m.notna()][['x_m','y_m']].to_numpy(float)
    low=moving.min(0)-4000; high=moving.max(0)+4000
    span=float(np.ceil(max(high-low)/4000)*4000); mid=(low+high)/2
    moving_bounds=(mid[0]-span/2,mid[0]+span/2,mid[1]-span/2,mid[1]+span/2)
    support_counts=[]
    for ax,key in zip(axes,(6775,6901,6998)):
        map_axis(ax,date(key),moving_bounds)
        rows=points[(points.image_id==str(key))&points.trajectory_id.isin(seed_ids)&points.x_m.notna()].set_index('trajectory_id')
        locations={idx:row[['x_m','y_m']].to_numpy(float) for idx,row in rows.iterrows()}
        segments=[np.array([locations[a],locations[b]])/1000 for a,b in mesh_edges if a in locations and b in locations]
        if segments:ax.add_collection(LineCollection(segments,color=TEAL,lw=1.1,zorder=3))
        dots(ax,rows[['x_m','y_m']].to_numpy(float),size=30)
        support_counts.append(len(rows))
    done(fig,'06_actual_mesh_advection',f'The same {len(seeds)} seeded identities are shown at A, C and D, using current composed trajectories. Connections between initial 4 km neighbours are a visual aid. Later positions deform the mesh; unsupported points and their connections are absent. No displacement magnification or forward prediction.')

    # Current scientific deformation output, plotted on the exact source triangles.
    f=fields[6901,6998];cells=core['deformation'].deformation_from_field(f,6400.)
    xy=f.source_xy_m[f.available];tri=Delaunay(xy).simplices
    used=np.array([cell.triangle_index for cell in cells],int)
    total=np.array([cell.total_deformation_s_1*86400 for cell in cells])
    div=np.array([cell.divergence_s_1*86400 for cell in cells])
    visible=inside(np.array([[c.centroid_x_m,c.centroid_y_m] for c in cells]))
    used=used[visible];total=total[visible];div=div[visible]
    assert len(used)>0 and np.isfinite(total).all()
    # Verify selected gradients independently by recovering velocities at vertices.
    gradient_errors=[]
    for t,rate in zip(used[:20],total[:20]):
        verts=xy[tri[t]];d=f.displacement_m[f.available][tri[t]]
        g=np.linalg.solve(np.column_stack((verts,np.ones(3))),d)/((f.target_time_utc-f.source_time_utc).total_seconds())
        gradient_errors.append(abs(np.hypot(g[0,0]+g[1,1],np.hypot(g[0,0]-g[1,1],g[1,0]+g[0,1]))*86400-rate))
    triang=Triangulation(xy[:,0]/1000,xy[:,1]/1000,tri[used])
    total_limit=max(.01,float(np.ceil(np.percentile(total,98)*100)/100))
    div_limit=max(.01,float(np.ceil(np.percentile(np.abs(div),98)*100)/100))
    fig,axes=page(7,'Deformation','')
    axes[0].set_position([.065,.13,.365,.71]);axes[1].set_position([.57,.13,.365,.71])
    for panel,(ax,values,title,cmap,norm) in enumerate(zip(axes,(total,div),('Total · C → D','Divergence · C → D'),
            ('magma','RdBu_r'),(Normalize(0,total_limit),TwoSlopeNorm(0,-div_limit,div_limit)))):
        map_axis(ax,title)
        artist=ax.tripcolor(triang,facecolors=values,cmap=cmap,norm=norm,edgecolors='none')
        cax=fig.add_axes([.445 if panel==0 else .947,.245,.012,.49])
        cbar=fig.colorbar(artist,cax=cax,extend='max' if title.startswith('Total') else 'both')
        cbar.ax.set_title('day⁻¹',size=10,pad=12);cbar.ax.tick_params(labelsize=9)
    done(fig,'07_actual_deformation','Rates from the current deformation_from_field implementation on valid source-field triangles for 30 April–2 May. EPSG:3413, maximum edge 6.4 km, orientation preserved. Total = hypot(divergence, shear). Colour limits round the 98th percentile upwards; out-of-range colours saturate. SIC masking is disabled in this diagnostic run; these are unmasked diagnostics.')

    # Reproduce every selected extension using the exact current baseline/fallback.
    checks=[]
    for pair_id,rows in extensions.groupby('pair_id'):
        a,b=map(int,pair_id.split('__'));edge=core['field'].field_edge_with_fallback(fields[a,b],matches[a,b],cfg)
        start=rows[['x0_m','y0_m']].to_numpy(float);end=rows[['x1_m','y1_m']].to_numpy(float)
        sampled=core['field'].sample_field_with_fallback(edge.field,edge.fallback_field,start,6400.)
        assert sampled.available.all();checks.extend(np.linalg.norm(start+sampled.displacement_m-end,axis=1))
    assert max(checks)<1e-5 and consensus_error<1e-5 and max(gradient_errors)<1e-10
    selections={'raw_match_rows':selected_vectors,'raw_detail_rows':marked,'consensus_node':node,
                'consensus_neighbours':neighbours,'consensus_agree':agree,'deformation_triangles':tri[used]}
    np.savez_compressed(out/'display_selections.npz',**selections)
    manifest=dict(dataset='N-ICE2015 imagery; current ELoFTR setup rerun 2026-10-02',data=str(data),
        source_commit=run['source_commit'],crs='EPSG:3413',bounds_m=BOUNDS,
        display='Actual geometry, 1× displacement; plain backgrounds after establishing imagery.',
        checks={'pair_checksums_verified':len(fields),'measured_extensions_verified':len(checks),
                'maximum_extension_error_m':float(max(checks)),'consensus_error_m':consensus_error,
                'deformation_gradient_error_day_1':float(max(gradient_errors)),
                'contributions_at_C':contribution_counts,'seed_count':len(seeds),
                'supported_at_A_C_D':support_counts},figures=RECORDS,
        caveat='Bounded sector, six diagnostic incoming pairs; Mac MPS eager execution. Not a full catalogue production run.',
        reproduction={'script':str(Path(__file__).resolve()),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      'arguments':sys.argv[1:]})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    cards=''.join(f'<section><a href="{r["name"]}.svg"><img src="{r["name"]}.png"></a><details><summary>Figure notes</summary><p>{r["caption"]}</p></details></section>' for r in RECORDS)
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>LiMOSAT ELoFTR workflow</title><style>body{max-width:1200px;margin:30px auto;background:#f6f8fa;color:#142d3c;font:15px system-ui;padding:0 20px}img{width:100%}section{margin:45px 0}details{margin:8px 0}p{line-height:1.5}</style><h1>ELoFTR workflow</h1><p>28 April–2 May 2015 · current setup</p>'+cards)
    thumbs=[]
    for r in RECORDS:
        im=Image.open(out/(r['name']+'.png')).convert('RGB');im.thumbnail((600,338));thumbs.append(im)
    overview=Image.new('RGB',(1800,3*338),BG)
    for i,im in enumerate(thumbs):overview.paste(im,((i%3)*600,(i//3)*338))
    overview.save(out/'00_series_overview.png')
    if PDF_EXPORT is not None:PDF_EXPORT.close();PDF_EXPORT=None
    print(json.dumps(manifest['checks'],indent=2))


if __name__=='__main__':main()
