nnstreamer_sources += files(
  'gsttensor_aggregator.c',
  'gsttensor_converter.c',
  'gsttensor_crop.c',
  'gsttensor_debug.c',
  'gsttensor_decoder.c',
  'gsttensor_demux.c',
  'gsttensor_if.c',
  'gsttensor_merge.c',
  'gsttensor_mux.c',
  'gsttensor_rate.c',
  'gsttensor_repo.c',
  'gsttensor_reposink.c',
  'gsttensor_reposrc.c',
  'gsttensor_sink.c',
  'gsttensor_sparsedec.c',
  'gsttensor_sparseenc.c',
  'gsttensor_sparseutil.c',
  'gsttensor_split.c',
  'gsttensor_transform.c',
  'gsttensor_trainer.c'
)

# gsttensorsrc
if tensor_src_iio_build
  nnstreamer_sources += files('gsttensor_srciio.c')
endif

# gsttensortransform
if orcc_support_is_available
  # tensor_transform does not need to generate -dist files.
  orcsrc = 'nnstreamer-orc'

  orc_h = custom_target(orcsrc + '.h',
    input: orcsrc + '.orc',
    output: orcsrc + '.h',
    command: orcc_support_orcc_args + ['--header', '-o', '@OUTPUT@', '@INPUT@'])
  orc_c_org = custom_target(orcsrc + '.c.in',
    input: orcsrc + '.orc',
    output: orcsrc + '.c.in',
    command: orcc_support_orcc_args + ['--implementation', '-o', '@OUTPUT@', '@INPUT@'])

  # ORC generated codes incur a few warnings. Ignore warnings of generated files.
  script = find_program('ignore_warning.sh')

  if build_platform == 'windows'
    script_path = script.path()
    drive_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    foreach letter : drive_letters
      windows_prefix = letter + ':' + '/'
      if script_path.startswith(windows_prefix)
        # str.replace() is available in Meson 0.58, but want to keep the minimum required version as
        # close as possible to the project's minimum required version.
        # Also, only need to replace the start of paths here.
        script_path = '/' + letter.to_lower() + '/' + script_path.substring(3)
      endif
    endforeach

    orc_c = custom_target(orcsrc + '.c',
      input: orc_c_org,
      output: orcsrc + '.c',
      command: [ ['bash', script_path], '@INPUT@', '@OUTPUT@' ])
  else
    orc_c = custom_target(orcsrc + '.c',
      input: orc_c_org,
      output: orcsrc + '.c',
      command: [ script, '@INPUT@', '@OUTPUT@' ])
  endif

  nnstreamer_sources += [orc_c, orc_h]
endif
