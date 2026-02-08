#!/bin/bash
# process-all-planetmath.sh — Clone all PlanetMath MSC repos and extract terms
#
# Clones each repo into ~/code/planetmath/<name>/
# Extracts terms from all .tex files
# Builds expanded NER kernel at the end
#
# Usage: bash scripts/process-all-planetmath.sh

set -euo pipefail

PM_DIR="$HOME/code/planetmath"
FUTON6_DIR="$HOME/code/futon6"
LOG_FILE="$FUTON6_DIR/data/planetmath-batch.log"
TERMS_DIR="$FUTON6_DIR/data/pm-all-terms"

mkdir -p "$PM_DIR" "$TERMS_DIR"

echo "=== PlanetMath Full Corpus Processing ===" | tee "$LOG_FILE"
echo "Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Get list of MSC repos
REPOS=$(gh repo list planetmath --limit 200 --json name -q '.[].name' | grep '^[0-9]' | sort)
TOTAL=$(echo "$REPOS" | wc -l)
echo "Found $TOTAL MSC repos" | tee -a "$LOG_FILE"

CLONED=0
SKIPPED=0
FAILED=0
TOTAL_TEX=0

for REPO in $REPOS; do
    REPO_DIR="$PM_DIR/$REPO"

    if [ -d "$REPO_DIR" ] && [ "$(find "$REPO_DIR" -name '*.tex' 2>/dev/null | head -1)" != "" ]; then
        TEX_COUNT=$(find "$REPO_DIR" -name '*.tex' | wc -l)
        echo "[$((CLONED + SKIPPED + FAILED + 1))/$TOTAL] SKIP $REPO (already have $TEX_COUNT .tex files)" | tee -a "$LOG_FILE"
        SKIPPED=$((SKIPPED + 1))
        TOTAL_TEX=$((TOTAL_TEX + TEX_COUNT))
        continue
    fi

    echo "[$((CLONED + SKIPPED + FAILED + 1))/$TOTAL] CLONE $REPO..." | tee -a "$LOG_FILE"

    if gh repo clone "planetmath/$REPO" "$REPO_DIR" -- --depth 1 2>>"$LOG_FILE"; then
        TEX_COUNT=$(find "$REPO_DIR" -name '*.tex' 2>/dev/null | wc -l)
        echo "  -> $TEX_COUNT .tex files" | tee -a "$LOG_FILE"
        CLONED=$((CLONED + 1))
        TOTAL_TEX=$((TOTAL_TEX + TEX_COUNT))
    else
        echo "  -> FAILED to clone" | tee -a "$LOG_FILE"
        FAILED=$((FAILED + 1))
    fi

    # Small delay to be polite to GitHub
    sleep 0.5
done

echo "" | tee -a "$LOG_FILE"
echo "=== Clone Summary ===" | tee -a "$LOG_FILE"
echo "Cloned: $CLONED" | tee -a "$LOG_FILE"
echo "Skipped (already had): $SKIPPED" | tee -a "$LOG_FILE"
echo "Failed: $FAILED" | tee -a "$LOG_FILE"
echo "Total .tex files: $TOTAL_TEX" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Now extract terms from ALL repos
echo "=== Extracting terms from all repos ===" | tee -a "$LOG_FILE"

cd "$FUTON6_DIR"
python3 << 'PYEOF'
import re, json, sys
sys.path.insert(0, 'src')
from futon6.latex_terms import extract_terms, extract_xrefs
from pathlib import Path
from collections import Counter

pm_root = Path('/home/joe/code/planetmath')

pat_canonical = re.compile(r'\\pmcanonicalname\{([^}]+)\}')
pat_title = re.compile(r'\\pmtitle\{([^}]+)\}')
pat_type = re.compile(r'\\pmtype\{([^}]+)\}')
pat_defines = re.compile(r'\\pmdefines\{([^}]+)\}')
pat_synonym = re.compile(r'\\pmsynonym\{([^}]*)\}\{([^}]*)\}')
pat_related = re.compile(r'\\pmrelated\{([^}]+)\}\{([^}]+)\}')
pat_msc = re.compile(r'\\pmclassification\{msc\}\{([^}]+)\}')
pat_body = re.compile(r'\\begin\{document\}(.*?)(?:\\end\{document\}|$)', re.DOTALL)

STOP_TERMS = {
    'the', 'remark', 'remarks', 'examples', 'definition', 'proof', 'via',
    'set', 'note', 'see', 'example', 'references', 'bibliography',
    'properties', 'introduction', 'background', 'notation', 'then',
}

def is_junk(t):
    t = t.strip()
    if len(t) < 3 or len(t) > 100:
        return True
    if re.match(r'^\d+$', t) or re.match(r'^\(\d+\)', t):
        return True
    if t.lower() in STOP_TERMS:
        return True
    if t.startswith('$') or t.startswith('\\') or t.startswith('"'):
        return True
    if any(x in t for x in ['Springer', 'Verlag', 'Academic Press', 'Oxford',
                             'Cambridge', 'Wiley', 'translation by', 'vol.',
                             'pp.', 'ISBN', 'edition', 'Bulletin of',
                             'Journal of', 'Annals of', 'Handbook of',
                             'Cahiers', 'Trans. Amer.', 'Proc.']):
        return True
    return False

all_terms = {}  # term_lower -> {term, sources, canonicals, types, msc_codes, domains}
all_entries = []  # for stats
domain_stats = Counter()
type_stats = Counter()
total_tex = 0
total_defines = 0
total_synonyms = 0
total_xrefs = 0

for d in sorted(pm_root.iterdir()):
    if not d.is_dir() or not d.name[0].isdigit():
        continue

    domain = d.name
    tex_count = 0

    for tex_file in sorted(d.glob('*.tex')):
        raw = tex_file.read_text(errors='replace')
        tex_count += 1
        total_tex += 1

        m = pat_canonical.search(raw)
        canonical = m.group(1) if m else None

        m = pat_title.search(raw)
        title = m.group(1) if m else None

        m = pat_type.search(raw)
        entry_type = m.group(1) if m else None

        msc_codes = [m.group(1) for m in pat_msc.finditer(raw)]

        if entry_type:
            type_stats[entry_type] += 1

        all_entries.append({
            'canonical': canonical,
            'title': title,
            'type': entry_type,
            'domain': domain,
            'msc_codes': msc_codes,
        })

        def add_term(term, source):
            global total_defines, total_synonyms
            t = term.strip()
            if is_junk(t):
                return
            key = t.lower()
            if key not in all_terms:
                all_terms[key] = {
                    'term': t,
                    'sources': set(),
                    'canonicals': set(),
                    'types': set(),
                    'domains': set(),
                    'msc_codes': set(),
                }
            all_terms[key]['sources'].add(source)
            if canonical:
                all_terms[key]['canonicals'].add(canonical)
            if entry_type:
                all_terms[key]['types'].add(entry_type)
            all_terms[key]['domains'].add(domain)
            for mc in msc_codes:
                all_terms[key]['msc_codes'].add(mc)

        if title and not is_junk(title):
            add_term(title, 'title')

        for m in pat_defines.finditer(raw):
            add_term(m.group(1), 'pmdefines')
            total_defines += 1

        for m in pat_synonym.finditer(raw):
            syn = m.group(1).strip()
            if syn:
                add_term(syn, 'pmsynonym')
                total_synonyms += 1

        body_m = pat_body.search(raw)
        body = body_m.group(1) if body_m else ''
        for term in extract_terms(body):
            add_term(term, 'body_emph')

        xrefs = extract_xrefs(body)
        total_xrefs += len(xrefs)

    domain_stats[domain] = tex_count
    if tex_count > 0:
        print(f'  {domain}: {tex_count} .tex files', file=sys.stderr)

# Build output
dict_entries = []
for key in sorted(all_terms):
    t = all_terms[key]
    structured = t['sources'] & {'title', 'pmdefines', 'pmsynonym'}
    confidence = 'high' if structured else 'medium'
    dict_entries.append({
        'term': t['term'],
        'term_lower': key,
        'sources': sorted(t['sources']),
        'defined_in': sorted(t['canonicals']),
        'entry_types': sorted(t['types']),
        'domains': sorted(t['domains']),
        'msc_codes': sorted(t['msc_codes']),
        'confidence': confidence,
    })

high = sum(1 for e in dict_entries if e['confidence'] == 'high')
med = sum(1 for e in dict_entries if e['confidence'] == 'medium')

print(f'\n=== Full PlanetMath Extraction ===', file=sys.stderr)
print(f'Total .tex files processed: {total_tex}', file=sys.stderr)
print(f'Total entries: {len(all_entries)}', file=sys.stderr)
print(f'Domains: {len(domain_stats)}', file=sys.stderr)
print(f'Total pmdefines: {total_defines}', file=sys.stderr)
print(f'Total pmsynonyms: {total_synonyms}', file=sys.stderr)
print(f'Total xrefs extracted: {total_xrefs}', file=sys.stderr)
print(f'Unique terms: {len(dict_entries)} ({high} high, {med} medium)', file=sys.stderr)
print(f'', file=sys.stderr)
print(f'Entry types:', file=sys.stderr)
for t, c in type_stats.most_common():
    print(f'  {t}: {c}', file=sys.stderr)
print(f'', file=sys.stderr)
print(f'Top 20 domains by .tex count:', file=sys.stderr)
for d, c in domain_stats.most_common(20):
    print(f'  {c:4d}  {d}', file=sys.stderr)

# Save full dictionary
out_dir = Path('data/pm-all-terms')
out_dir.mkdir(exist_ok=True)

with open(out_dir / 'pm-full-dictionary.json', 'w') as f:
    json.dump(dict_entries, f, indent=1, ensure_ascii=False)

# Save TSV for bb
with open(out_dir / 'pm-full-terms.tsv', 'w') as f:
    f.write('term_lower\tterm\tconfidence\tdomains\tdefined_in\n')
    for e in dict_entries:
        domains = ';'.join(e['domains'][:5])
        defined = ';'.join(e['defined_in'][:3])
        f.write(f"{e['term_lower']}\t{e['term']}\t{e['confidence']}\t{domains}\t{defined}\n")

# Save domain stats
with open(out_dir / 'domain-stats.json', 'w') as f:
    json.dump({
        'total_tex': total_tex,
        'total_entries': len(all_entries),
        'total_terms': len(dict_entries),
        'high_confidence': high,
        'medium_confidence': med,
        'total_defines': total_defines,
        'total_synonyms': total_synonyms,
        'total_xrefs': total_xrefs,
        'domains': dict(domain_stats.most_common()),
        'entry_types': dict(type_stats.most_common()),
    }, f, indent=2)

print(f'\nSaved to {out_dir}/', file=sys.stderr)
print(f'  pm-full-dictionary.json', file=sys.stderr)
print(f'  pm-full-terms.tsv', file=sys.stderr)
print(f'  domain-stats.json', file=sys.stderr)
PYEOF

echo "" | tee -a "$LOG_FILE"
echo "=== Now rebuilding NER kernel with full corpus ===" | tee -a "$LOG_FILE"

# Rebuild the NER kernel with the full PlanetMath + SE
bb scripts/build-ner-kernel.bb --se-json data/se-physics.json 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "=== Completed ===" | tee -a "$LOG_FILE"
echo "Finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG_FILE"
