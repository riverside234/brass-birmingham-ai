param([int]$SeedCount=1000000)
$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
$dll=Join-Path $root 'BrassBirminghamUnity_Data/Managed/Assembly-CSharp.dll'
$hash=(Get-FileHash -LiteralPath $dll -Algorithm SHA256).Hash
if($hash -ne '600EBF309670B2726001C16A3375F8BBF3DE76A23C0202BE7CF198BA338981FB'){throw 'Assembly mismatch'}
[void][Reflection.Assembly]::LoadFrom($dll)
function New-IdentityGame([int]$seed){
 $cfg=[BoardGameRules.Entities.Game.GameStartConfiguration]::new([BoardGameRules.Entities.Common.NumberOfPlayers]::Four)
 $cfg.Seed=$seed
 $g=[BoardGameRules.Entities.Game.BoardGameRulesGame]::new($cfg)
 $g.StartGame()
 return $g
}
$baseline=New-IdentityGame 1
$draws=$baseline.Randomer.RandomerInitData.FastForwardCount
Add-Type -TypeDefinition @'
using System;
using System.Collections.Generic;
public static class AuditIdentitySeeds {
 public static int[] Candidates(int count, int draws) {
  var hits=new List<int>(); var set=new HashSet<int>();
  for(int seed=0;seed<count;seed++) {
   var rng=new Random(seed); set.Clear();
   for(int i=0;i<draws;i++) if(!set.Add(rng.Next())) { hits.Add(seed); break; }
  }
  return hits.ToArray();
 }
}
'@
$candidates=[AuditIdentitySeeds]::Candidates($SeedCount,$draws)
$collisions=[Collections.Generic.List[object]]::new()
foreach($seed in $candidates){
 $g=New-IdentityGame $seed
 $items=[Collections.Generic.List[object]]::new()
 foreach($p in $g.PlayersTrack.Players){
  foreach($type in @('CoalMine','IronWorks','Pottery','Brewery','CottonMill','Manufacturer')){
   foreach($t in $p.Mat.GetIndustryList([BoardGameRules.Entities.Board.BuildingSlots.BuildingType]::$type)){$items.Add([pscustomobject]@{kind='industry';id=$t.UniqueId;color=$p.Color.ToString();type=$type;level=$t.Level;tile=$t})}
  }
 }
 foreach($r in $g.Board.Regions){
  foreach($s in $r.BuildingSlots){$items.Add([pscustomobject]@{kind='slot';id=$s.UniqueId;region=$r.Name.ToString()})}
  if($r -is [BoardGameRules.Entities.Board.Regions.BorderRegion]){foreach($m in $r.MerchantSlots){$items.Add([pscustomobject]@{kind='merchant';id=$m.UniqueId;region=$r.Name.ToString()})}}
 }
 $cards=@($g.Board.CardDeck)+@($g.Board.WildIndustryCardDeck)+@($g.Board.WildLocationCardDeck)+@($g.PlayersTrack.Players|ForEach-Object {$_.Hand;$_.DiscardPile})
 foreach($c in $cards){$items.Add([pscustomobject]@{kind='card';id=$c.UniqueId;type=$c.GetType().Name})}
 $dupes=@($items|Group-Object id|Where-Object Count -gt 1)
 foreach($dup in $dupes){
  $entries=@($dup.Group|Select-Object kind,id,color,type,level,region)
  $lookups=@($dup.Group|Where-Object kind -eq 'industry'|ForEach-Object {[pscustomobject]@{requestedColor=$_.color;requestedType=$_.type;resolvedColor=$g.GetIndustryByUniqueId($_.id).Color.ToString();resolvedType=$g.GetIndustryByUniqueId($_.id).BuildingType.ToString();sameObject=[object]::ReferenceEquals($_.tile,$g.GetIndustryByUniqueId($_.id))}})
  $collisions.Add([ordered]@{seed=$seed;entries=$entries;lookupResults=$lookups})
 }
}
$result=[ordered]@{assemblySha256=$hash;runtime=[Runtime.InteropServices.RuntimeInformation]::FrameworkDescription;method='Scan seeded System.Random outputs as a prefilter, then construct original unmodified four-player games and check industry, card, merchant, and building-slot identities. No board-state mutations.';seedsScanned=$SeedCount;drawsPerSeed=$draws;candidateSeeds=$candidates.Count;collisions=@($collisions.ToArray())}
$result|ConvertTo-Json -Depth 12|Set-Content -LiteralPath (Join-Path $PSScriptRoot 'extended-identity-results.json') -Encoding utf8
[ordered]@{seeds=$SeedCount;draws=$draws;candidates=$candidates.Count;collisions=$collisions.Count}|ConvertTo-Json
