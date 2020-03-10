echo "this is test"
printf 'what I read from stdin: %s\n' "$(cat)"
{ printf 'what I read from stdout: %s\n' "$(cat <&3)"; } 3<&1
printf 'what I read from stderr: %s\n' "$(cat <&2)"

echo HOSTNAME=$(hostname) NODENUM=$PBS_NODENUM VNODENUM=$PBS_VNODENUM

