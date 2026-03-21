"""Step 7: Export graphs to NetworkX formats and interactive HTML visualization."""
import json
import math
import os
import pickle
from collections import defaultdict
from typing import Dict, Any, List

import networkx as nx

from src.shared.logger import logger


class NetworkXExporter:
    """Exports entity and experience graphs using NetworkX."""

    def __init__(self, output_dir: str = "data/networkx"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def export_entity_graph(
        self,
        experiences: List[Dict[str, Any]],
        output_prefix: str = "entity_graph",
        lexicon_path: str = "data/lexicon/lexicon.json",
        entity_postings_path: str = "data/entity_postings/entity_postings.json",
    ) -> None:
        """Build and export entity graph from entity_edges.

        Entity names are canonicalized via lexicon (alias→canonical mapping from
        Step 2.5). exp_ids per entity are loaded from entity_postings.json (Step 4)
        which is already keyed by canonical names.
        """
        logger.log_info(f"\n{'='*80}")
        logger.log_info("Exporting Entity Graph to NetworkX")
        logger.log_info(f"{'='*80}")

        # Load lexicon and postings built in Steps 2.5 and 4.
        alias_to_canonical = self._load_alias_mapping(lexicon_path)
        entity_postings = self._load_entity_postings(entity_postings_path)
        logger.log_info(f"  Lexicon aliases: {len(alias_to_canonical)}")
        logger.log_info(f"  Entity postings: {len(entity_postings)} canonical entities")

        g = nx.MultiDiGraph()

        # Build canonical → role by scanning ALL core_entities (not just edge entities).
        # This covers isolated entities that never appear in entity_edges.
        canonical_roles: Dict[str, str] = {}
        for exp in experiences:
            for item in exp.get("core_entities", []):
                ent = item.get("entity")
                role = item.get("role")
                if ent and role:
                    canonical = self._canonicalize_entity(ent, alias_to_canonical)
                    if canonical not in canonical_roles:
                        canonical_roles[canonical] = role

        # Add ALL canonical entities from entity_postings as nodes (4,702 total).
        # Isolated entities (not in any edge) are included with degree 0.
        for entity, exp_ids in entity_postings.items():
            role = canonical_roles.get(entity, "Unknown")
            g.add_node(entity, role=role, exp_ids=exp_ids)

        # Add edges with canonicalized entity names.
        # Skip edges whose endpoints are not in entity_postings (rare LLM artifacts).
        for exp in experiences:
            exp_id = exp.get("id", "")
            for e_edge in exp.get("entity_edges", []):
                edge_type = e_edge.get("edge", "")
                src = self._canonicalize_entity(
                    e_edge.get("from_entity", ""), alias_to_canonical)
                dst = self._canonicalize_entity(
                    e_edge.get("to_entity", ""), alias_to_canonical)
                if src in g and dst in g:
                    g.add_edge(src, dst, edge_type=edge_type, exp_id=exp_id)

        self._save_graph_outputs(g, output_prefix)
        self._save_html_visualization(g, f"{output_prefix}_interactive.html", "entity")

        logger.log_info(f"  Entity graph nodes: {g.number_of_nodes()}")
        logger.log_info(f"  Entity graph edges: {g.number_of_edges()}")

    def export_experience_graph(
        self,
        experiences: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        output_prefix: str = "experience_graph"
    ) -> None:
        """Build and export experience similarity graph."""
        logger.log_info(f"\n{'='*80}")
        logger.log_info("Exporting Experience Graph to NetworkX")
        logger.log_info(f"{'='*80}")

        g = nx.Graph()

        exp_lookup = {exp["id"]: exp for exp in experiences if "id" in exp}
        for exp_id, exp in exp_lookup.items():
            source = exp.get("source", "static")
            g.add_node(
                exp_id,
                task_type=exp.get("task_type_norm", exp.get("task_type", "unknown")),
                condition=exp.get("condition", "")[:300],
                content=exp.get("content", "")[:300],
                quality=exp.get("quality"),
                source=source,
            )

        for edge in edges:
            src = edge.get("src")
            dst = edge.get("dst")
            if not src or not dst:
                continue
            g.add_edge(
                src, dst,
                W=float(edge.get("W", 0.0)),
                S_ent=float(edge.get("S_ent", 0.0)),
                S_graph=float(edge.get("S_graph", 0.0)),
                S_sem=float(edge.get("S_sem", 0.0)),
                S_task=float(edge.get("S_task", 0.0)),
                reason=edge.get("short_reason", "")
            )

        self._save_graph_outputs(g, output_prefix)
        self._save_html_visualization(g, f"{output_prefix}_interactive.html", "experience")

        logger.log_info(f"  Experience graph nodes: {g.number_of_nodes()}")
        logger.log_info(f"  Experience graph edges: {g.number_of_edges()}")

    def _load_alias_mapping(self, lexicon_path: str) -> Dict[str, str]:
        """Load alias→canonical mapping from Step 2.5 lexicon."""
        if not os.path.exists(lexicon_path):
            logger.log_warning(f"Lexicon not found: {lexicon_path}. Using raw entity names.")
            return {}
        try:
            with open(lexicon_path, "r", encoding="utf-8") as f:
                lexicon = json.load(f)
            return lexicon.get("alias_to_canonical", {})
        except Exception as e:
            logger.log_warning(f"Failed to load lexicon: {e}")
            return {}

    def _load_entity_postings(self, postings_path: str) -> Dict[str, List[str]]:
        """Load entity→exp_ids mapping from Step 4 entity_postings.json."""
        if not os.path.exists(postings_path):
            logger.log_warning(f"Entity postings not found: {postings_path}.")
            return {}
        try:
            with open(postings_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.log_warning(f"Failed to load entity postings: {e}")
            return {}

    def _canonicalize_entity(self, entity: str, alias_to_canonical: Dict[str, str]) -> str:
        """Map raw entity name to canonical form (mirrors Step 5b logic)."""
        if not entity:
            return ""
        if entity in alias_to_canonical:
            return alias_to_canonical[entity]
        normalized = " ".join(entity.strip().lower().split())
        return alias_to_canonical.get(normalized, normalized)

    def _save_graph_outputs(self, graph: nx.Graph, output_prefix: str) -> None:
        """Save graph as pickle and node-link JSON."""
        gpickle_path = os.path.join(self.output_dir, f"{output_prefix}.gpickle")
        with open(gpickle_path, "wb") as f:
            pickle.dump(graph, f)

        json_path = os.path.join(self.output_dir, f"{output_prefix}.json")
        data = nx.node_link_data(graph)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.log_info(f"  Graph pickle: {gpickle_path}")
        logger.log_info(f"  Node-link JSON: {json_path}")

    def _save_html_visualization(
        self, graph: nx.Graph, filename: str, graph_type: str
    ) -> None:
        """Generate interactive HTML visualization using vis.js Network."""
        if graph.number_of_nodes() == 0:
            return

        out_path = os.path.join(self.output_dir, filename)
        vis_nodes, vis_edges = [], []
        node_info, edge_info = {}, {}

        degrees = dict(graph.degree())
        max_deg = max(degrees.values()) if degrees else 1

        if graph_type == "experience":
            title = "GSEM Experience Graph"
            palette = {
                "diagnosis": "#4A90D9", "treatment": "#50C878",
                "ttl_diagnosis": "#FF8C42", "ttl_treatment": "#E05C4F",
                "ttl_unknown": "#FF8C42",
            }
            borders = {
                "diagnosis": "#357ABD", "treatment": "#3DAF63",
                "ttl_diagnosis": "#CC6A1E", "ttl_treatment": "#B03A2E",
                "ttl_unknown": "#CC6A1E",
            }

            for nid, attrs in graph.nodes(data=True):
                tt = attrs.get("task_type", "unknown")
                src = attrs.get("source", "static")
                deg = degrees.get(nid, 0)
                sz = 5 + math.log1p(deg) / math.log1p(max_deg) * 20
                color_key = f"ttl_{tt}" if src == "ttl" else tt
                bg = palette.get(color_key, "#B0BEC5")
                bd = borders.get(color_key, "#90A4AE")
                vis_nodes.append({
                    "id": nid,
                    "label": nid if src == "ttl" else nid.replace("exp_", ""),
                    "size": round(sz, 1),
                    "shape": "diamond" if src == "ttl" else "dot",
                    "color": {
                        "background": bg,
                        "border": bd,
                        "highlight": {"background": bg, "border": bd},
                    },
                })
                node_info[nid] = {
                    "task_type": tt, "degree": deg,
                    "quality": attrs.get("quality"),
                    "source": src,
                    "condition": attrs.get("condition", ""),
                    "content": attrs.get("content", ""),
                }

            eid = 0
            for src, dst, attrs in graph.edges(data=True):
                W = attrs.get("W", 0.0)
                w_norm = max(0, (W - 0.35)) / 0.65
                vis_edges.append({
                    "id": eid, "from": src, "to": dst,
                    "width": round(0.4 + w_norm * 4, 2),
                    "color": {"color": "#848484",
                              "opacity": round(0.12 + w_norm * 0.7, 3)},
                })
                edge_info[str(eid)] = {
                    "W": round(W, 4),
                    "S_ent": round(attrs.get("S_ent", 0.0), 4),
                    "S_graph": round(attrs.get("S_graph", 0.0), 4),
                    "S_sem": round(attrs.get("S_sem", 0.0), 4),
                    "S_task": round(attrs.get("S_task", 0.0), 4),
                    "reason": attrs.get("reason", ""),
                }
                eid += 1

            legend_html = (
                '<div class="lg"><span class="dot" style="background:#4A90D9"></span>Diagnosis</div>'
                '<div class="lg"><span class="dot" style="background:#50C878"></span>Treatment</div>'
                '<div class="lg"><span class="dot" style="background:#FF8C42;border-radius:2px;transform:rotate(45deg)"></span>TTL Diagnosis</div>'
                '<div class="lg"><span class="dot" style="background:#E05C4F;border-radius:2px;transform:rotate(45deg)"></span>TTL Treatment</div>'
            )
            options = {
                "nodes": {
                    "shape": "dot",
                    "font": {"size": 0, "color": "#333",
                             "strokeWidth": 3,
                             "strokeColor": "rgba(255,255,255,0.85)"},
                    "borderWidth": 1.5,
                },
                "edges": {
                    "smooth": {"type": "continuous", "roundness": 0.15},
                    "color": {"inherit": False},
                    "selectionWidth": 2,
                },
                "physics": {
                    "barnesHut": {
                        "gravitationalConstant": -1500,
                        "centralGravity": 0.8,
                        "springLength": 60,
                        "springConstant": 0.04,
                        "damping": 0.09,
                        "avoidOverlap": 0.1,
                    },
                    "stabilization": {
                        "enabled": True, "iterations": 300,
                        "updateInterval": 25,
                    },
                    "maxVelocity": 50, "minVelocity": 0.1,
                },
                "interaction": {
                    "hover": True, "tooltipDelay": 300,
                    "hideEdgesOnDrag": True,
                },
            }
            detail_js = _EXP_DETAIL_JS

        else:  # entity
            title = "GSEM Entity Graph"
            palette = {
                "Condition": "#FF8C42", "Constraint": "#E74C3C",
                "Action": "#4A90D9", "Rationale": "#9B59B6",
                "Outcome": "#2ECC71", "Unknown": "#95A5A6",
            }
            borders = {
                "Condition": "#E07630", "Constraint": "#C0392B",
                "Action": "#357ABD", "Rationale": "#8E44AD",
                "Outcome": "#27AE60", "Unknown": "#7F8C8D",
            }

            for nid, attrs in graph.nodes(data=True):
                role = attrs.get("role", "Unknown")
                deg = degrees.get(nid, 0)
                sz = 4 + math.log1p(deg) / math.log1p(max_deg) * 18
                lbl = nid if len(nid) <= 20 else nid[:18] + "\u2026"
                vis_nodes.append({
                    "id": nid, "label": lbl,
                    "size": round(sz, 1),
                    "color": {
                        "background": palette.get(role, "#95A5A6"),
                        "border": borders.get(role, "#7F8C8D"),
                        "highlight": {
                            "background": palette.get(role, "#95A5A6"),
                            "border": borders.get(role, "#7F8C8D"),
                        },
                    },
                })
                node_info[nid] = {
                    "role": role, "degree": deg,
                    "exp_ids": attrs.get("exp_ids", []),
                }

            eid = 0
            for src, dst, _key, attrs in graph.edges(data=True, keys=True):
                vis_edges.append({
                    "id": eid, "from": src, "to": dst,
                    "arrows": "to", "width": 1,
                    "color": {"color": "#848484", "opacity": 0.4},
                })
                edge_info[str(eid)] = {
                    "edge_type": attrs.get("edge_type", ""),
                    "exp_id": attrs.get("exp_id", ""),
                }
                eid += 1

            legend_html = "".join(
                f'<div class="lg"><span class="dot" style="background:{palette[r]}"></span>{r}</div>'
                for r in ["Condition", "Constraint", "Action", "Rationale", "Outcome"]
            )
            options = {
                "nodes": {
                    "shape": "dot",
                    "font": {"size": 0, "color": "#333",
                             "strokeWidth": 3,
                             "strokeColor": "rgba(255,255,255,0.85)"},
                    "borderWidth": 1.5,
                },
                "edges": {
                    "arrows": {"to": {"enabled": True, "scaleFactor": 0.5}},
                    "smooth": {"type": "continuous", "roundness": 0.15},
                    "color": {"inherit": False},
                    "selectionWidth": 2,
                },
                "physics": {
                    "barnesHut": {
                        "gravitationalConstant": -2000,
                        "centralGravity": 0.6,
                        "springLength": 80,
                        "springConstant": 0.04,
                        "damping": 0.09,
                        "avoidOverlap": 0.05,
                    },
                    "stabilization": {
                        "enabled": True, "iterations": 200,
                        "updateInterval": 25,
                    },
                    "maxVelocity": 50, "minVelocity": 0.1,
                },
                "interaction": {
                    "hover": True, "tooltipDelay": 300,
                    "hideEdgesOnDrag": True,
                },
            }
            detail_js = _ENT_DETAIL_JS

            # Build experience → entities reverse mapping for dual-mode search.
            exp_to_entities: Dict[str, List[str]] = defaultdict(list)
            for entity, info in node_info.items():
                for eid in info.get("exp_ids", []):
                    exp_to_entities[eid].append(entity)
            exp2ent_json = json.dumps(dict(exp_to_entities), ensure_ascii=False)

        # Assemble HTML
        html = _HTML_TEMPLATE
        replacements = {
            "__TITLE__": title,
            "__N_NODES__": str(graph.number_of_nodes()),
            "__N_EDGES__": str(graph.number_of_edges()),
            "__LEGEND__": legend_html,
            "__NODES__": json.dumps(vis_nodes, ensure_ascii=False),
            "__EDGES__": json.dumps(vis_edges, ensure_ascii=False),
            "__NODE_INFO__": json.dumps(node_info, ensure_ascii=False),
            "__EDGE_INFO__": json.dumps(edge_info, ensure_ascii=False),
            "__OPTIONS__": json.dumps(options, ensure_ascii=False),
            "__DETAIL_JS__": detail_js,
            "__EXP2ENT__": exp2ent_json if graph_type == "entity" else "{}",
            "__SEARCH_PH__": "Search entity or exp_XXXX\u2026" if graph_type == "entity" else "Search node\u2026",
        }
        for placeholder, value in replacements.items():
            html = html.replace(placeholder, value)

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html)
        logger.log_info(f"  Interactive HTML: {out_path}")


# ---------------------------------------------------------------------------
# Detail-panel JS builders (experience vs entity)
# ---------------------------------------------------------------------------
_EXP_DETAIL_JS = """
function buildNodeDetail(id, d) {
    var src = d.source || 'static';
    var taskColor = src === 'ttl'
        ? (d.task_type==='diagnosis'?'#FF8C42':'#E05C4F')
        : (d.task_type==='diagnosis'?'#4A90D9':'#50C878');
    var dot = '<span class="td" style="background:' + taskColor + '"></span>';
    var srcBadge = src === 'ttl'
        ? '<span style="font-size:9px;font-weight:700;color:#fff;background:#FF8C42;'
          + 'border-radius:3px;padding:1px 5px;margin-left:4px;vertical-align:middle">TTL</span>'
        : '';
    var q = (d.quality !== null && d.quality !== undefined) ? d.quality.toFixed(3) : 'N/A';
    return row('ID', id + srcBadge, 1)
         + row('Task Type', dot + d.task_type)
         + row('Quality (Q)', q)
         + row('Connections', d.degree)
         + row('Condition', esc(d.condition))
         + row('Content', esc(d.content));
}
function buildEdgeDetail(e, d) {
    return row('Connection', e.from + ' \\u2194 ' + e.to, 1)
         + '<div class="sh">Similarity Scores</div>'
         + sb('W', d.W||0, '#333')
         + sb('S_ent', d.S_ent||0, '#FF8C42')
         + sb('S_graph', d.S_graph||0, '#9B59B6')
         + sb('S_sem', d.S_sem||0, '#2ECC71')
         + sb('S_task', d.S_task||0, '#E74C3C')
         + (d.reason ? row('Reason', esc(d.reason)) : '');
}
"""

_ENT_DETAIL_JS = """
function buildNodeDetail(id, d) {
    var ids = d.exp_ids || [];
    var expHtml = ids.map(function(e){
        return '<span style="display:inline-block;font-family:monospace;font-size:10px;'
             + 'background:#f0f4ff;border:1px solid #d0d8f0;border-radius:3px;'
             + 'padding:1px 5px;margin:2px 2px 0 0">' + e + '</span>';
    }).join('');
    return row('Entity', id)
         + row('Role', d.role || 'Unknown')
         + row('Connections', d.degree)
         + row('Appears in', ids.length + ' experience' + (ids.length===1?'':'s'))
         + (ids.length ? '<div class="rw"><div class="lb">Experience IDs</div>'
             + '<div style="max-height:140px;overflow-y:auto;margin-top:2px">'
             + expHtml + '</div></div>' : '');
}
function buildEdgeDetail(e, d) {
    return row('Connection', e.from + ' \\u2192 ' + e.to, 1)
         + row('Edge Type', d.edge_type || '', 1)
         + row('Experience', d.exp_id || '', 1);
}
"""

# ---------------------------------------------------------------------------
# Complete HTML template
# ---------------------------------------------------------------------------
_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<script src="https://unpkg.com/vis-network@9.1.6/standalone/umd/vis-network.min.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
     background:#f8f9fa;color:#333;height:100vh;overflow:hidden}
#app{display:flex;flex-direction:column;height:100vh}
#hdr{display:flex;align-items:center;justify-content:space-between;
     padding:10px 20px;background:#fff;border-bottom:1px solid #e5e5e5}
#hdr h1{font-size:15px;font-weight:600}
.st{font-size:12px;color:#888}
#tb{display:flex;align-items:center;gap:8px;padding:6px 20px;
    background:#fff;border-bottom:1px solid #eee}
.btn{padding:4px 10px;font-size:11px;border:1px solid #ddd;border-radius:4px;
     background:#fff;cursor:pointer;color:#555;transition:all .15s}
.btn:hover{background:#f0f0f0;border-color:#bbb}
.btn.on{background:#4A90D9;color:#fff;border-color:#4A90D9}
#main{display:flex;flex:1;overflow:hidden}
#gc{flex:1;position:relative}
#gv{width:100%;height:100%}
#pn{width:320px;background:#fff;border-left:1px solid #e5e5e5;
    overflow-y:auto;display:flex;flex-direction:column}
#ph{padding:12px 16px;font-size:12px;font-weight:600;color:#888;
    text-transform:uppercase;letter-spacing:.5px;border-bottom:1px solid #eee}
#pc{padding:14px 16px;flex:1;font-size:13px;line-height:1.5}
.hint{color:#bbb;text-align:center;padding:40px 0;font-size:13px}
.rw{margin-bottom:10px}
.lb{font-size:10px;font-weight:600;color:#aaa;text-transform:uppercase;
    letter-spacing:.3px;margin-bottom:2px}
.vl{color:#333;word-break:break-word}
.mn{font-family:'SF Mono',Menlo,monospace;font-size:12px}
.td{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px;
    vertical-align:middle}
.sh{margin:12px 0 8px;font-size:10px;font-weight:600;color:#aaa;
    text-transform:uppercase;letter-spacing:.3px}
.sb{display:flex;align-items:center;gap:4px;margin-bottom:4px}
.sb-l{width:50px;font-size:10px;color:#888}
.sb-t{flex:1;height:5px;background:#eee;border-radius:3px;overflow:hidden}
.sb-f{height:100%;border-radius:3px;transition:width .3s}
.sb-v{width:42px;font-size:10px;color:#555;text-align:right;font-family:monospace}
#lg{padding:12px 16px;border-top:1px solid #eee}
#lg h3{font-size:10px;font-weight:600;color:#aaa;text-transform:uppercase;
       letter-spacing:.5px;margin-bottom:6px}
.lg{display:flex;align-items:center;gap:6px;margin-bottom:3px;font-size:12px}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%}
#ld{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);text-align:center}
.spn{width:28px;height:28px;border:2px solid #e0e0e0;border-top-color:#4A90D9;
     border-radius:50%;animation:sp .8s linear infinite;margin:0 auto 10px}
@keyframes sp{to{transform:rotate(360deg)}}
#sr{padding:3px 8px;font-size:11px;border:1px solid #ddd;border-radius:4px;
    width:150px;outline:none}
#sr:focus{border-color:#4A90D9;box-shadow:0 0 0 2px rgba(74,144,217,.15)}
</style>
</head>
<body>
<div id="app">
  <div id="hdr">
    <h1>__TITLE__</h1>
    <span class="st">__N_NODES__ nodes &middot; __N_EDGES__ edges</span>
  </div>
  <div id="tb">
    <button class="btn" onclick="network.fit({animation:{duration:400}})">Fit View</button>
    <button class="btn on" id="bp" onclick="togPhy()">Physics: ON</button>
    <button class="btn" id="bl" onclick="togLbl()">Labels: OFF</button>
    <input type="text" id="sr" placeholder="__SEARCH_PH__" oninput="doSearch(this.value)">
  </div>
  <div id="main">
    <div id="gc">
      <div id="gv"></div>
      <div id="ld"><div class="spn"></div><div style="font-size:13px;color:#666">Stabilizing… <span id="pct">0%</span></div></div>
    </div>
    <div id="pn">
      <div id="ph">Details</div>
      <div id="pc"><div class="hint">Click a node or edge<br>to view details</div></div>
      <div id="lg"><h3>Legend</h3>__LEGEND__</div>
    </div>
  </div>
</div>
<script>
var ND=__NODES__,ED=__EDGES__,NI=__NODE_INFO__,EI=__EDGE_INFO__,E2N=__EXP2ENT__;
var nodes=new vis.DataSet(ND),edges=new vis.DataSet(ED);
var network=new vis.Network(document.getElementById('gv'),
    {nodes:nodes,edges:edges},__OPTIONS__);
var phyOn=true,lblOn=false;

// --- stabilization ---
network.on('stabilizationProgress',function(p){
    document.getElementById('pct').textContent=Math.round(p.iterations/p.total*100)+'%';
});
network.on('stabilizationIterationsDone',function(){
    document.getElementById('ld').style.display='none';
});

// --- click: detail + neighborhood highlight ---
network.on('click',function(p){
    var el=document.getElementById('pc');
    if(p.nodes.length){
        var id=p.nodes[0],d=NI[id];
        // highlight neighborhood
        var nb=new Set(network.getConnectedNodes(id));nb.add(id);
        var ce=new Set(network.getConnectedEdges(id));
        nodes.update(ND.map(function(n){return{id:n.id,opacity:nb.has(n.id)?1:.08}}));
        edges.update(ED.map(function(e){
            return{id:e.id,color:{color:e.color.color||'#848484',
                   opacity:ce.has(e.id)?0.7:0.02}};
        }));
        if(d) el.innerHTML=buildNodeDetail(id,d);
    }else if(p.edges.length){
        resetHL();
        var eid=p.edges[0],e=edges.get(eid),d=EI[String(eid)]||{};
        if(e) el.innerHTML=buildEdgeDetail(e,d);
    }else{
        resetHL();
        el.innerHTML='<div class="hint">Click a node or edge<br>to view details</div>';
    }
});
network.on('doubleClick',function(p){
    if(p.nodes.length) network.focus(p.nodes[0],{scale:2.5,animation:{duration:500}});
});

function resetHL(){
    nodes.update(ND.map(function(n){return{id:n.id,opacity:1}}));
    edges.update(ED.map(function(e){return{id:e.id,color:e.color}}));
}

// --- helpers ---
function esc(s){return s?String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'):''}
function row(l,v,m){return'<div class="rw"><div class="lb">'+l+'</div><div class="vl'+(m?' mn':'')+'">'+v+'</div></div>'}
function sb(l,v,c){var p=Math.round(v*100);
    return'<div class="sb"><div class="sb-l">'+l+'</div><div class="sb-t"><div class="sb-f" style="width:'+p+'%;background:'+c+'"></div></div><div class="sb-v">'+v.toFixed(3)+'</div></div>';}

__DETAIL_JS__

// --- controls ---
function togPhy(){
    phyOn=!phyOn;network.setOptions({physics:{enabled:phyOn}});
    var b=document.getElementById('bp');
    b.textContent='Physics: '+(phyOn?'ON':'OFF');b.className=phyOn?'btn on':'btn';
}
function togLbl(){
    lblOn=!lblOn;
    nodes.update(ND.map(function(n){return{id:n.id,font:{size:lblOn?11:0}}}));
    var b=document.getElementById('bl');
    b.textContent='Labels: '+(lblOn?'ON':'OFF');b.className=lblOn?'btn on':'btn';
}
function doSearch(q){
    if(!q){resetHL();return;}
    var qt=q.trim();
    // Experience mode: input matches exp_XXXX and exists in E2N
    if(E2N[qt]){
        var entSet=new Set(E2N[qt]);
        // Highlight nodes belonging to this experience
        nodes.update(ND.map(function(n){return{id:n.id,opacity:entSet.has(n.id)?1:.06};}));
        // Highlight edges whose exp_id matches this experience
        var expEdges=new Set();
        ED.forEach(function(e){if(EI[String(e.id)]&&EI[String(e.id)].exp_id===qt)expEdges.add(e.id);});
        edges.update(ED.map(function(e){
            var hit=expEdges.has(e.id);
            return{id:e.id,color:{color:hit?'#4A90D9':'#848484',opacity:hit?.85:.02},width:hit?2:1};
        }));
        network.selectNodes(E2N[qt].filter(function(id){return!!nodes.get(id);}));
        var chips=E2N[qt].map(function(e){
            return '<span onclick="focusEntity(\''+e.replace(/\\/g,'\\\\').replace(/'/g,"\\'")+'\') " '
                 + 'style="display:inline-block;font-size:10px;font-family:monospace;cursor:pointer;'
                 + 'background:#f0f4ff;border:1px solid #d0d8f0;border-radius:3px;'
                 + 'padding:1px 5px;margin:2px 2px 0 0">'+esc(e)+'</span>';
        }).join('');
        document.getElementById('pc').innerHTML=
            row('Experience',qt,1)+row('Entities',E2N[qt].length)
            +'<div class="rw"><div class="lb">Entity List</div>'
            +'<div style="max-height:200px;overflow-y:auto;margin-top:2px">'+chips+'</div></div>';
        return;
    }
    // Entity name mode
    var ql=qt.toLowerCase();var hit=null;
    nodes.update(ND.map(function(n){
        var m=n.id.toLowerCase().indexOf(ql)>=0||(n.label&&n.label.toLowerCase().indexOf(ql)>=0);
        if(m&&!hit)hit=n.id;return{id:n.id,opacity:m?1:.08};
    }));
    if(hit)network.focus(hit,{scale:1.8,animation:{duration:400}});
}
function focusEntity(id){
    resetHL();
    network.focus(id,{scale:2,animation:{duration:400}});
    network.selectNodes([id]);
}
</script>
</body>
</html>
"""
