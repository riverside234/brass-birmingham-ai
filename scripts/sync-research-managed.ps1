# Keep isolated research players independent of a newer installed UI/AI patch.
# Dot-source this file; it never stages into the installed game's Managed folder.
function Sync-BrassResearchManaged {
    param([string]$ProjectRoot,[string]$Source,[string]$Destination)
    $artifactRoot=[IO.Path]::GetFullPath((Join-Path $ProjectRoot 'artifacts')).TrimEnd('\','/')+[IO.Path]::DirectorySeparatorChar
    $target=[IO.Path]::GetFullPath($Destination).TrimEnd('\','/')
    $sourcePath=[IO.Path]::GetFullPath($Source).TrimEnd('\','/')
    if(-not $target.StartsWith($artifactRoot,[StringComparison]::OrdinalIgnoreCase) -or
       -not $target.EndsWith(('BrassBirminghamUnity_Data'+[IO.Path]::DirectorySeparatorChar+'Managed'),[StringComparison]::OrdinalIgnoreCase) -or
       $sourcePath.StartsWith($target+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase) -or
       $sourcePath -eq $target){throw 'Research DLL staging requires a separate isolated player below project artifacts.'}
    $directory=Get-Item -LiteralPath $target -ErrorAction Stop
    for($ancestor=$directory;$ancestor -and $ancestor.FullName.StartsWith($artifactRoot.TrimEnd('\','/'),[StringComparison]::OrdinalIgnoreCase);$ancestor=$ancestor.Parent){
        if($ancestor.Attributes -band [IO.FileAttributes]::ReparsePoint){throw 'Research DLL staging cannot traverse a directory link.'}
    }
    $files=@(Get-ChildItem -LiteralPath $sourcePath -Filter '*.dll' -File -ErrorAction Stop)
    foreach($required in 'Assembly-CSharp.dll','BrassPatch.dll'){
        if($required -notin $files.Name){throw "Missing research dependency: $required"}
    }
    # Only DLLs in the disposable player copy are replaced/removed. This also
    # removes an installed 1.3 deployment DLL from an R1/research-EXE test copy.
    $existing=@(Get-ChildItem -LiteralPath $target -Filter '*.dll' -File)
    if(@($existing|Where-Object {$_.Attributes -band [IO.FileAttributes]::ReparsePoint}).Count){throw 'Research DLL staging cannot replace file links.'}
    foreach($file in $existing){
        if($file.Name -notin $files.Name){Remove-Item -LiteralPath $file.FullName -Force}
    }
    foreach($file in $files){
        $destinationFile=Join-Path $target $file.Name
        Copy-Item -LiteralPath $file.FullName -Destination $destinationFile -Force
        if((Get-FileHash -LiteralPath $destinationFile).Hash -ne (Get-FileHash -LiteralPath $file.FullName).Hash){
            throw "Research DLL staging mismatch: $($file.Name)"
        }
    }
}
