remotetolocalport="5800"

alias root-color-on="sudo sed -i -e 's/#force_color_prompt=yes/force_color_prompt=yes/g' ~/.bashrc"
alias root-color-off="sudo sed -i -e 's/force_color_prompt=yes/#force_color_prompt=yes/g' ~/.bashrc"

root-color-on

CHECK_ALIASING=$(cat ~/.bashrc | grep -w '^source.*alias.bash$')
if [ -z "$CHECK_ALIASING" ]; then
	echo "source ~/alias.bash" >> ~/.bashrc
fi

# install -> i
alias ipy3="sudo apt-get update && sudo apt install python3-pip -y && iipython"
alias ipipenv="pip install pipenv"
alias icowsay="sudo apt install cowsay -y"
alias iftpserver='sudo apt install vsftpd -y'
alias isshserver='sudo apt-get install openssh-server -y'
alias iapache2server="sudo apt install apache2 -y && sudo ufw allow \"Apache full\" && sudo apt install -y php libapache2-mod-php php-mysql php-curl php-gd php-mbstring php-zip php-json php-xml && sudo /etc/init.d/apache2 restart"
alias itmux="sudo apt-get install -y tmux"
alias iunzip="sudo apt install -y unzip"
alias ilibtmux="pip install --user libtmux"
alias iipython="pip install ipython"
alias ipysftp="pip install pysftp"
alias igdown="pip install gdown"
alias iprettyerrors="pip install pretty_errors"
alias itorch="pip install torch"
alias ipl="pip install pytorch-lightning"
alias ierd='sudo apt install graphviz && apt install graphviz-dev && pipenv install pyparsing pydot && pipenv install django-extensions'

# high level customization
CD ()
{
    eval "mkdir -p $1 && cd $1"
}
RM ()
{
    eval "sudo rm -rf $1"
}

# kernel -> k
alias kwhat="type -a"
alias kos="uname"
alias kv="uname -r"
alias kV="cat /proc/version"
alias knet="nmcli"
alias knet-advanced="sudo tcpdump -D"
alias knet-device="nmcli device status"
alias knet-condev="knet-device | grep connected | kf-split 1"
alias knet-tc-advanced="sudo iptraf-ng"
alias kcpu="cat /proc/cpuinfo"
alias khost="hostnamectl"
alias kcg-tree="systemd-cgls"
alias kcg-controllers="cat /proc/cgroups"
alias kcg-controllers-advanced="lssubsys"
alias kcg-controllers-advanced-paths="lssubsys -M"
alias kcg-controllers-advanced-paths-only="kcg-controllers-advanced-paths | grep '\/.*' -o | grep ''"
alias kmemory-ls="free -m"
ksplit ()
{
	set -- "${1:-}" "${2:-1G}" "${3:-$(openssl rand -hex 12)}"
	local oldpath=$(pwd)
	eval "mkdir -p $3"
	eval "cd $3"
	eval "split ../$1 -b$2"
	eval "cd $oldpath"
}
alias kmerge="cat ./* > merged"
alias kmerge-ckpt="kmerge && unzip ./merged"
alias kpo="fwop"
alias kpc="fwcp"
kf-pls ()
{
	set -- "${1:-}" "${2:-ltnp}"
	eval "sudo netstat -$2 | grep -w '.*$1' | grep ''"
}
kf-split ()
{
	set -- "${1:-0}" "${2:- }"
	while read -r line; do
        	local data="$data $line"
    	done
	local data=$(echo $data | tr '\n' ' ' | xargs)
	if [ "$1" -eq "0" ]; then
   		echo $data;
   		exit;
	fi
	echo $(echo $data | cut -d"$2" -f$1)
}
kf-entering ()
{
	while read -r line; do
        	local data="$data $line"
    	done
	eval "echo \"$data\" | xargs | sed -e 's/\s/\n/g'"
}
kf-entering-numbers ()
{
	while read -r line; do
        	local data="$data $line"
    	done
	eval "echo \"$data\" | xargs | sed -e 's/\s/\n/g' | grep -n ''"
}
kf-select ()
{
	set -- "${1:-1}"
	while read -r line; do
        	local data="$data $line"
    	done
	eval "echo \"$data\" | xargs | sed -e 's/\s/\n/g' | grep -n '' | grep -w '^$1:.*' | sed -e 's/^$1\://g'"
}

# notify
alias alert='notify-send --urgency=low -i "$([ $? = 0 ] && echo terminal || echo error)" "$(history|tail -n1|sed -e '\''s/^\s*[0-9]\+\s*//;s/[;&|]\s*alert$//'\'')"'
dm ()
{
	set -- "${1:-hey}"
	targetfile="/tmp/.custom_pushbask_notification.bash"
	echo -e "while true; do echo -ne \"__backslash__a\"; sleep 1; done &\nFOO_PID=\$!\nif zenity --info --text \"\$1\"; then\nkill \$FOO_PID\necho \"yes\"\nelse\nkill \$FOO_PID\necho \"no\"\nfi" | sed -e 's/__backslash__/\\/g' > "$targetfile"
	bash "$targetfile" "$1"
}

# alias
alias py="python3 -c 'import IPython; IPython.terminal.ipapp.launch_new_instance()'"
alias ipython="py"
alias pipf="pip freeze > Requirements.txt"
alias alias-f="declare"
alias alias-F="declare -F"
alias alias-g="alias | grep"
alias-def ()
{
    eval "alias $1=\"$2\""
}

# grep
alias grep='grep --color=auto'
alias egrep='egrep --color=auto'
alias fgrep='fgrep --color=auto'

# list
alias l='ls -CF'
alias la='ls -A'
alias ll='ls -alFh'
alias ls='ls --color=auto'

# link
alias hln="ln" # hln src dst
alias sln="ln -sf" # sln src dst

# filesystem
alias mkp='mkdir -p'

# ps
alias ps1='ps -p 1 -o comm='

# user -> u
alias uadd='sudo useradd -m '

# systemctl -> sc
alias sc='sudo systemctl'
alias scs='sudo systemctl status '
alias scd='sudo systemctl disable '
alias sce='sudo systemctl enable '
alias scr='sudo systemctl restart '
alias scstart='sudo systemctl start '
alias scstop='sudo systemctl stop '
alias scfSD='sudo systemctl --type=service --state=dead'
alias scfSE='sudo systemctl --type=service --state=exited'
alias scfSF='sudo systemctl --type=service --state=failed'
alias scfSI='sudo systemctl --type=service --state=inactive'
alias scfSR='sudo systemctl --type=service --state=running'

# service -> ss | Reserved: ss
alias ssrftp='ssr vsftpd'
ssr ()
{
    echo $(sudo service $1 restart)
}
sss ()
{
    echo $(sudo service $1 status)
}
ssstart ()
{
    echo $(sudo service $1 start)
}
ssstop ()
{
    echo $(sudo service $1 stop)
}

# firewall -> fw
alias fwcp='sudo ufw deny '
alias fwop='sudo ufw allow '

# apache2
alias apache2r="sudo /etc/init.d/apache2 restart"

# ftp
alias ftph='ftp $(hostname)'
alias scstartftp='scstart vsftpd'
alias sceftp='sce vsftpd'
alias scrftp='scr vsftpd'
wgetftp ()
{
	set -- "${1:-/}" "${2:-/}" "${3:-localhost}" "${4:-root}" "${5:-toor}"
	eval "wget -r \"ftp://$3/$1\" --ftp-user=\"$4\" --ftp-password=\"$5\" -P \"$2\""
}
pemsftp ()
{
        set -- "${1:-}" "${2:-~/VPS/ac-fum-privateKey.pem}" "${3:-5.160.40.35}" "${4:-ubuntu}"
        eval "sudo sftp -P $1 -i $2 $4@$3"
}

# ssh
alias scstartssh='scstart ssh'
alias scessh='sce ssh'
alias rsapubssh="sudo cat ~/.ssh/id_rsa.pub"
dmssh ()
{
	local msg1=$(echo "$1" | sed -e 's/\s/_/g')
	eval "curl --silent localhost:$remotetolocalport/notify/$USER@$(hostname)/$msg1"
}
pemssh ()
{
	set -- "${1:-}" "${2:-}" "${3:-}" "${4:-}"
	eval "sudo scp -P $1 -i $2 ~/alias.bash $4@$3:~/alias.bash"
	eval "sudo ssh -p $1 -i $2 $4@$3 \"sudo ln -sf ~/alias.bash /root/alias.bash && sudo fuser \"$remotetolocalport/tcp\" -k\""
	eval "sudo ssh -R $remotetolocalport:localhost:$remotetolocalport -i $2 $4@$3 -p $1"
}
pemssh-fum ()
{
	set -- "${1:-}" "${2:-~/VPS/fum/ali.pem}" "${3:-5.160.40.35}" "${4:-ubuntu}"
	eval "pemssh $1 $2 $3 $4"
}
pemssh-download ()
{
	set -- "${1:-}" "${2:-~/test.txt}" "${3:-~/test.txt}" "${4:-~/VPS/fum/ali.pem}" "${5:-5.160.40.35}" "${6:-ubuntu}"
	eval "sudo scp -P $1 -i $4 $6@$5:$2 $3"
}

# npm
alias npm-driver="sudo apt install -y nodejs npm && sudo npm install pm2 -g && pm2 completion install"

# gdown
gdown-folder ()
{
	eval "gdown --folder $1 && dmssh \"download completed\" &"
}

# projects -> P
alias Pinit="iprettyerrors && iipython && igdown" # run inside venv (isolated)
alias Pgmli="Pinit && wget 'https://raw.githubusercontent.com/mvlabfum/ML_Framework/main/index.bash' -O - | bash /dev/stdin"
alias Pgml="wget 'https://raw.githubusercontent.com/mvlabfum/ML_Framework/main/index.bash' -O - | bash /dev/stdin"
alias Pgsi="Pinit && ..." # TODO
alias Pgs="..." # TODO
Pmake ()
{
	set -- "${1:-$(openssl rand -hex 12)}" "${2:-pl}" "${3:-}" "${4:-net}"
	
	if [ -d "$3apps/${1^^}" ]; then
		eval "rm -rf $3apps/${1^^}/configs/-"
		eval "rm -rf $3apps/${1^^}/configs/${4,,}"
		eval "cp -R $3templates/app/$2/configs/- $3apps/${1^^}/configs/${4,,}"
	else
		eval "cp -R $3templates/app/$2 $3apps/${1^^}"
		eval "mv \"$3apps/${1^^}/configs/-\" \"$3apps/${1^^}/configs/${4,,}\""
	fi

	eval "rm -rf $3apps/${1^^}/configs/-"

	if [ $2 = "pl" ]; then
		sed -i s/{{@APP}}/"${1^^}"/g $3apps/${1^^}/*.py
		sed -i s/{{@APP}}/"${1^^}"/g $3apps/${1^^}/modules/*.py
		sed -i s/{{@APP}}/"${1^^}"/g $3apps/${1^^}/modules/**/*.py
		sed -i s/{{@APP}}/"${1^^}"/g $3apps/${1^^}/models/*.py
		sed -i s/{{@APP}}/"${1^^}"/g $3apps/${1^^}/data/*.py
		
		sed -i s/{{@APP}}/"${1^^}"/g $3apps/${1^^}/configs/${4,,}/*.yaml
		sed -i s/{{@APP}}/"${1^^}"/g $3apps/${1^^}/configs/${4,,}/.yaml
		sed -i s/{{@NET}}/"${4,,}"/g $3apps/${1^^}/configs/${4,,}/*.yaml
		sed -i s/{{@NET}}/"${4,,}"/g $3apps/${1^^}/configs/${4,,}/.yaml
	fi
}
Pmake-colab ()
{
	set -- "${1:-$(openssl rand -hex 12)}" "${2:-net}" "${3:-GENIE_ML}" "${4:-pl}"
	local uppath="/content/$3/"
	eval "Pmake $1 $4 $uppath $2"
	local pwdvar=$(pwd)
	eval "cd $uppath"
	local git_full_token=$(cat /content/.git_full_token)
	eval "gupload \"Created using Colaboratory\" c $git_full_token"
	eval "cd $pwdvar"
}

alias Pvenv-ls="sudo ls /root/.local/share/virtualenvs/"
Pvenv ()
{
	set -- "${1:-$(openssl rand -hex 12)}"
	local pwdvar=$(pwd)
	local var1=$(echo "$1" | xargs | sed -e 's/\s/_/g')
	echo "$var1"
	eval "sudo mkdir -p \"/content/devs/$var1\" && cd \"/content/devs/$var1\""
	sudo echo -e "[[source]]\nurl = \"https://pypi.python.org/simple\"\nverify_ssl = true \nname = \"pypi\"\n\n[dev-packages]\n\n[scripts]\n\n[requires]\npython_version=\"3.8\"\n\n[packages]" > Pipfile
	eval "sudo pipenv install && sudo pipenv shell"
	cd $pwdvar
}

# tmux -> T
alias Tserver="tmux start-server"
alias Tkserver="tmux kill-server"
alias Tls="tmux list-session"
alias Tcr-without-s="tmux new-session -d"
alias Tcr="tmux new-session -d -s " # -s session_name | optional: -n window_name
alias Tup="tmux attach-session -t " # session_name
alias Tdn=""
alias Tdel="tmux kill-session -t " # session_name
alias Trename="tmux rename-session -t" # current_session_name new_session_name

# mysql
alias scstartmysql='scstart mysql.service'

# s3
s3-upload ()
{
	set -- "${1:-}" "${2:-}" "${3:-}" "${4:-}" "${5:-}"
	local S3KEYS=$(cat "$4" | grep "$5"_"$3")
	local S3KEYS_AK=$(echo $S3KEYS | kf-entering | grep -w '.*ak' | sed -e s/.*ak\://g)
	local S3KEYS_SK=$(echo $S3KEYS | kf-entering | grep -w '.*sk' | sed -e s/.*sk\://g)
	local S3KEYS_EURL=$(echo $S3KEYS | kf-entering | grep -w '.*eurl' | sed -e s/.*eurl\://g)
	local RANDOMsTRING=$(openssl rand -hex 12)
	local py_filepath="/tmp/s3_upload_custom_$RANDOMsTRING.py"
	echo -e "import tqdm, os, boto3; s3 = boto3.resource('s3', endpoint_url=\"$S3KEYS_EURL\", aws_access_key_id=\"$S3KEYS_AK\", aws_secret_access_key=\"$S3KEYS_SK\"); fpath=\"$1\"; file_size = os.stat(fpath).st_size;\nwith tqdm.tqdm(total=file_size, unit='B', unit_scale=True, desc=fpath) as pbar: s3.Bucket(\"$3\").upload_file(Filename=fpath, Key=\"$2\" or os.path.split(fpath)[1], Callback=lambda bytes_transferred: pbar.update(bytes_transferred),)" > "$py_filepath"
	python3 "$py_filepath"
	sudo rm -rf "$py_filepath"
}
s3-upload-liara ()
{
	set -- "${1:-~}" "${2:-''}" "${3:-serverfum}" "${4:-~/credentials/liara-hbpserver}" "${5:-storage}"
	eval "s3-upload $1 $2 $3 $4 $5"
}

# git -> g | Reserved -> alias gt='ssh -T git@gitlab.com'
alias g='ga && gc'
alias ga='git add .'
alias gb='git checkout '
alias gbD='git branch -D '
alias gbd='git branch -d '
alias gbds='git push -d '
alias gbf='git checkout -b'
alias gbv='git branch -v'
alias gbva='git branch -a'
alias gc='git commit -m '
alias gcc='git commit --amend -m '
alias gclear='git rm -r --cached '
alias gg='ga && gcc'
alias gl='git log'
alias gl1='git log --oneline'
alias glr='git remote -v'
alias glt='git log --branches --remotes --tags --graph --oneline --decorate'
alias gp='git push'
alias gpull='git pull'
alias gr1='git reset --hard HEAD~1'
alias grh='git reset --hard '
alias grs='git reset --soft '
gupload () 
{ 
	set -- "${1:-'bugfix'}" "${2:-'h'}" "${3:-'master'}"
	eval "g '$1'"

	if [[ $2 == *"h"* ]]; then
		eval "gp github $3"
	fi
	if [[ $2 == *"l"* ]]; then
		eval "gp gitlab $3"
	fi
	if [[ $2 == *"o"* ]]; then
		eval "gp origin $3"
	fi
	if [[ $2 == *"c"* ]]; then
		eval "gp $3"
	fi
}
gmerge () 
{ 
    echo $(grh $1 && gcc $2)
}

# django(python manage) -> pm
alias pm='python manage.py'
alias pm2m='python manage.py makemigrations'
alias pmapp='python manage.py startapp '
alias pmcs='python manage.py collectstatic --noinput'
alias pmdbsh='python manage.py dbshell'
alias pmdd='python manage.py dumpdata'
alias pmerddot='python manage.py graph_models -a > documentation/ERD.dot'
alias pmerd='pmerddot && python manage.py graph_models --pydot -a -g -o documentation/ERD.png'
alias pmdoc='mkp documentation && pmerd'
alias pmld='python manage.py loaddata'
alias pmm='python manage.py migrate'
alias pmmg='pm2m && python manage.py migrate'
alias pmrs='python manage.py runserver'
alias pmsh='python manage.py shell'
alias pmsm='python manage.py sqlmigrate auth 0001'
alias pmsync='python manage.py syncdb --noinput'
alias pmt='python manage.py test'
alias pmu='python manage.py createsuperuser'
alias pmurp='python manage.py changepassword '
pmseed () 
{ 
    echo $(python manage.py seed $1 --number=$2)
}
pmsql () 
{ 
    echo $(python manage.py makemigrations $1 --empty)
}

# temp
alias tempopenportftp='fwop 20/tcp && fwop 21/tcp'

# task -> t
alias tftpserver='iftpserver && scstartftp && sceftp && tempopenportftp '
alias tftpserver-colab="iftpserver && /etc/init.d/vsftpd start"
alias tsshserver='isshserver && scessh && scstartssh'

# initialize vps -> init
alias init-vps="ipy3 && ipipenv && npm-driver && icowsay && tftpserver && iapache2server && itmux && iunzip && Pvenv" # you can determine that which env is used
alias init-fum="init-vps"
init-colab-i ()
{
	set -- "${1:-Pgmli Pi}"
	eval "icowsay && iunzip && $1"
}
init-colab ()
{
	set -- "${1:-Pgml Pi}"
	eval "$1"
}
