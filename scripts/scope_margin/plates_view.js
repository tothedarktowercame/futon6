const scopeData = JSON.parse(document.getElementById('m7-scopes').textContent);
const scopeHost = document.querySelector('.m7-scope-host');
const scopeSegment = scopeData.segments[0].id;
renderScopes(scopeData, scopeHost, {
  anchored: {[scopeSegment]: window.m7units},
  layers: ['s1-term', 's3', 's4'],
});

const removeScopeTitles = () => window.m7units.forEach(u => u.e.removeAttribute('title'));
scopeHost.addEventListener('change', removeScopeTitles);
removeScopeTitles();
const scopeLegend = scopeHost.querySelector('.scope-legend');
if (scopeLegend) scopeLegend.open = false;
