import pretty_errors
from os import getenv, system
from settings import opt_main # --main argument -> default is 'app'
from libs.basicIO import S3Base
from libs.dyimport import Import

opt_main = opt_main.replace('\'', '"')
opt_main0 = opt_main.split(':')[0]
opt_main1_end_str = ' '.join(opt_main.split(':')[1:])

if opt_main0 == 'app':
    main = Import(f'apps.{getenv("GENIE_ML_APP_MD")}.app.App', embedParams={})

if opt_main0 == 's3':
    if opt_main1_end_str == 'upload':
        main = lambda: print('TODO: S3Base class')

if opt_main0.endswith('.py'):
    system('python3 "-u" {} {}'.format(opt_main0, opt_main1_end_str))
    main = lambda: None