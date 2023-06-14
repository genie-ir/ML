# small letter -> consider
# caps letter -> ignore
# sudoproject is located at /content/devs/?
# repo it will be located at $projectpath/$projectname

envfile="env"
argfile="args"
projectpath="/content" # fell free to choose it any what you like
projectname="${5:-ML_Framework}" # (Notic) the name of project on the `repository`.
projectrepo="${4:-mvlabfum}"
projectrepo="https://github.com/$projectrepo/$projectname.git"

sudo mkdir -p "$projectpath"

APP="$2"
NET="$3"

if [ -z "$APP" ]; then
    APP="APP"
fi
if [ -z "$NET" ]; then
    NET="NET"
fi

if [[ $1 == *"G"* ]]; then
    eval ":"
else
    cd "$projectpath"
    sudo rm -rf "$projectpath/$projectname"

    cd "$projectpath"
    git clone "$projectrepo"
fi

cd "$projectpath/$projectname"

if [[ $1 == *"i"* ]]; then
    pip install -r Requirements.txt
fi

# python -m pytorch_lightning.utilities.upgrade_checkpoint --file "/content/drive/MyDrive/storage/Genie_ML/VQGAN/logs/2022-12-14T14-24-22_eyepacs_vqgan/checkpoints/last.ckpt"

SIMLINK_ROOTDIR=""
SIMLINK_STORAGEi=""
A=$(python3 -c 'import sys; print(sys.prefix)')
B="${2:-$projectname}"
if [[ -f "$A/.project" ]]; then
    C=$(cat $A/.project | xargs | sed -e s/' '//g)
    ln -sf "$projectpath/$projectname" "$C/Repository"
    SIMLINK_ROOTDIR=" --simlink-rootdir $C/Root" # space is needed at first
    SIMLINK_STORAGEi=" --simlink-storagei $C/Storage" # space is needed at first
else
    C="$projectpath"
fi

D="-t True\n--gpus 0,\n--app \"${APP^^}:${APP^^}\"\n--envp \"$C/$envfile\"\n--ckpt \"last\"\n--base \"${NET,,}\"\n-r \"\""
if [[ ! -f "$C/$argfile" ]]; then
    echo -e "$D" > "$C/$argfile"
fi
if [[ ! -f "$C/$envfile" ]]; then
    eval "cp $projectpath/$projectname/.env $C/$envfile"
fi

sed -i s/--app.*/"--app \"${APP^^}:${APP^^}\""/g "$C/$argfile"
sed -i s/--base.*/"--base \"${NET,,}\""/g "$C/$argfile"
E=$(cat "$C/$argfile" | sed -e s/[\'\"]/\|/g | xargs | sed -e s/\|/\"/g)
E="$E$SIMLINK_ROOTDIR$SIMLINK_STORAGEi"
if [[ $1 == *"P"* ]]; then
    F="python3 \"-u\" $projectpath/$projectname/index.py $E"
else
    F="pm2 start $projectpath/$projectname/index.py --name \"$B\" --interpreter \"$A/bin/python3\" -- $E"
fi

if [[ $1 == *"S"* ]]; then
    eval ":"
else
    printf %"$COLUMNS"s | tr " " "-"
    echo "$F"
    printf %"$COLUMNS"s | tr " " "-"
    eval "$F"
fi