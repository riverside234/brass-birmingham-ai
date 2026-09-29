// Audit-only: the sole external JS evaluated here is the fully reviewed, data-only
// gameData.js. No require/process/network/DOM is exposed to its VM context.
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const crypto = require('crypto');
const root = path.resolve(__dirname, '../..');
const referencePath = path.join(root, 'artifacts/reference-npow-brass/js/gameData.js');
const source = fs.readFileSync(referencePath, 'utf8');
const reference = vm.runInNewContext(source + '\n({ INDUSTRY_DATA, CITIES, MERCHANTS, MERCHANT_TILE_MIX, CONNECTIONS, CARD_DECK, COAL_MARKET_INITIAL, IRON_MARKET_INITIAL, INITIAL_MONEY, INITIAL_INCOME_SPACE, HAND_SIZE })', {}, {timeout: 1000});
const dump = JSON.parse(fs.readFileSync(path.join(__dirname, 'components-dll-data.json'), 'utf8').replace(/^\uFEFF/, ''));
const sorted = xs => xs.map(x => JSON.stringify(x)).sort();
let passed = 0;
const differences = [];
const compare = (label, actual, expected) => {
  if (JSON.stringify(actual) === JSON.stringify(expected)) passed++;
  else differences.push({label, dll: actual, reference: expected});
};
const industryFields = ['level','vp','linkVP','income','canDevelop','canalEra','railEra','cost','costCoal','costIron','beersToSell','resourceCubes'];
const industries = Object.entries(reference.INDUSTRY_DATA).flatMap(([type, entries]) => entries.flatMap(tile => Array.from({length:tile.count}, () => Object.fromEntries([['type',type],...industryFields.map(k => [k,tile[k]])]))));
const cards = players => {
  const d = reference.CARD_DECK[players];
  return Object.entries(d.locations).flatMap(([k,n])=>Array(n).fill('location:'+k))
    .concat(Object.entries(d.industries).flatMap(([k,n])=>Array(n).fill('industry:'+k)))
    .concat(Array(d.dualCottonManufacturer || 0).fill('industry:cottonMill+manufacturer'));
};
const links = reference.CONNECTIONS.flatMap(c=>['canal','rail'].filter(era=>c[era]).map(era=>({era,ends:c.cities.concat(c.viaBrewery?[c.viaBrewery]:[]).sort()})));
for (const c of dump.cases) {
  const prefix = c.players + 'p:';
  for (const tile of c.industries) {
    const target = reference.INDUSTRY_DATA[tile.type].find(t=>t.level===tile.level);
    for (const field of industryFields) compare(prefix+tile.type+':'+tile.level+':'+field, tile[field], target[field]);
  }
  compare(prefix+'industry-multiset', sorted(c.industries), sorted(industries));
  for (const [name, city] of Object.entries(reference.CITIES)) compare(prefix+'city:'+name, sorted(c.cities[name].map(s=>s.slice().sort())), sorted(city.slots.map(s=>s.slice().sort())));
  compare(prefix+'northern-farm-slot', c.cities.northern, [['brewery']]);
  compare(prefix+'southern-farm-slot', c.cities.southern, [['brewery']]);
  compare(prefix+'location-count', Object.keys(c.cities).length, Object.keys(reference.CITIES).length+2);
  compare(prefix+'link-multiset', sorted(c.links), sorted(links));
  compare(prefix+'card-multiset', c.cards.slice().sort(), cards(c.players).sort());
  for (const [name, m] of Object.entries(reference.MERCHANTS)) {
    for (const field of ['slots','minPlayers','bonusType']) compare(prefix+'merchant:'+name+':'+field,c.merchants[name][field],m[field]);
  }
  compare(prefix+'merchant-tile-mix', c.merchantTiles.map(t=>t.type).sort(), reference.MERCHANT_TILE_MIX[c.players].slice().sort());
  compare(prefix+'merchant-slot-count', c.merchantTiles.length, Object.values(reference.MERCHANTS).filter(m=>m.minPlayers<=c.players).reduce((a,m)=>a+m.slots,0));
  for (const m of c.merchantTiles) compare(prefix+'merchant-beer:'+m.location+':'+m.type,m.beer,m.type==='blank'?0:1);
  compare(prefix+'coal-market-start',c.coalMarket,reference.COAL_MARKET_INITIAL);
  compare(prefix+'iron-market-start',c.ironMarket,reference.IRON_MARKET_INITIAL);
  compare(prefix+'wild-industry-count',c.wildIndustry,c.players);
  compare(prefix+'wild-location-count',c.wildLocation,c.players);
  for (const p of c.playerSetup) {
    compare(prefix+'player-money',p.money,reference.INITIAL_MONEY);
    compare(prefix+'player-income-marker',p.incomeMarker,reference.INITIAL_INCOME_SPACE);
    compare(prefix+'player-hand',p.hand,reference.HAND_SIZE);
    compare(prefix+'player-discard',p.discard,1);
    compare(prefix+'player-link-tiles',p.linkTiles,14);
  }
}
const result = {
  assemblySha256:dump.assemblySha256,helperSha256:dump.helperSha256,
  referenceCommit:'fdeb63c6aa92cc4ba44a721482316e0b8db10e06',
  referenceFileSha256:crypto.createHash('sha256').update(source).digest('hex').toUpperCase(),
  runtime:dump.runtime,seed:dump.seed,passedComparisons:passed,differences,
  note:'Comparison against an independent implementation is a differential check, not independent official-rules proof. See components-findings.md for adjudication.'
};
fs.writeFileSync(path.join(__dirname,'components-comparison-results.json'),JSON.stringify(result,null,2)+'\n');
console.log(JSON.stringify({passedComparisons:passed,differences},null,2));
