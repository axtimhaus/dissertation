$pdf_mode = 4;
@default_files = ("dissertation/dissertation.tex");
ensure_path("TEXINPUTS", ".", ".build/");
$do_cd=1;
$silent_mode=1;