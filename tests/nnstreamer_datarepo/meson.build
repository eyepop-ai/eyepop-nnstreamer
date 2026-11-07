# Run unittest_datareposink
unittest_datareposink = executable('unittest_datareposink',
  'unittest_datareposink.cc',
  dependencies: [nnstreamer_unittest_deps],
  install: get_option('install-test'),
  install_dir: unittest_install_dir
)
# Both tests create and delete img_%02d.png, img.json and mnist.* in the
# build directory, so they must not run at the same time.
test('unittest_datareposink', unittest_datareposink, env: testenv, is_parallel: false)

# Run unittest_datareposrc
unittest_datareposrc = executable('unittest_datareposrc',
  'unittest_datareposrc.cc',
  dependencies: [nnstreamer_unittest_deps],
  install: get_option('install-test'),
  install_dir: unittest_install_dir
)
test('unittest_datareposrc', unittest_datareposrc, env: testenv, is_parallel: false)
