# Automatic Conductor Cloud access to the Lenovo

New yamkit cloud workspaces can join your tailnet automatically and SSH directly
to the Lenovo. Each workspace gets its own ephemeral identity named
`conductor-yamkit-cloud-<workspace suffix>`. No Mac command, local bridge, public
SSH port, or motor command is involved.

The one-time setup below supplies an OAuth enrollment credential to Conductor.
Unlike a reusable auth key, the OAuth credential can keep creating enrollment
keys without a 90-day auth-key replacement cycle. It grants permission to enroll
devices with one tag; it does not need device administration or policy editing.
See [Tailscale OAuth clients](https://tailscale.com/docs/features/oauth-clients).

## 1. Configure Tailscale access once

In your [Tailscale access policy](https://login.tailscale.com/admin/acls), merge
the following entries into the corresponding sections. Preserve unrelated rules.
Add these tag owners:

```json
"tagOwners": {
  "tag:conductor-yamkit": ["autogroup:owner", "autogroup:admin"],
  "tag:yam-lenovo": ["autogroup:owner", "autogroup:admin"]
}
```

Add this entry to `grants`:

```json
{
  "src": ["tag:conductor-yamkit"],
  "dst": ["tag:yam-lenovo"],
  "ip": ["tcp:22"]
}
```

If your policy still has the default allow-all grant with `"src": ["*"]`,
change that grant's source to `"src": ["autogroup:member"]`. Keep its other
fields. This preserves access for user-owned devices while keeping tagged
cloud agents under their specific grant. Grants are additive: another broad
grant or ACL matching the cloud tag would also allow its traffic.
See [grant evaluation](https://tailscale.com/docs/reference/syntax/grants).

Add these entries to `ssh`:

```json
{
  "action": "accept",
  "src": ["tag:conductor-yamkit"],
  "dst": ["tag:yam-lenovo"],
  "users": ["andre"]
},
{
  "action": "check",
  "src": ["autogroup:owner", "autogroup:admin"],
  "dst": ["tag:yam-lenovo"],
  "users": ["andre"]
}
```

The first rule gives cloud agents unattended SSH. The second preserves the tailnet
owner's and administrators' access to the Lenovo after it becomes a tagged device. Tagged
sources require a tagged SSH destination and cannot use browser check mode.
See [Tailscale SSH](https://tailscale.com/docs/features/tailscale-ssh).

After saving the policy, open Machines, select `yam-lenovo`, and use **Edit tags**
to assign `tag:yam-lenovo`. Keep Tailscale SSH enabled on the Lenovo. The Linux
account remains `andre`; tagging changes the Tailscale device identity.

For a rig that should remain reachable without periodic node-key login, inspect
its **Key expiry** setting on the Machines page. Disabling expiry for this
dedicated Lenovo is a separate choice; a tag added in the admin console does
not automatically change an existing expiry setting.
See [device tags](https://tailscale.com/docs/features/tags).

## 2. Create the enrollment credential once

Open [Trust credentials](https://console.tailscale.com/admin/settings/trust-credentials),
choose **Credential → OAuth**, and grant **Auth Keys → Write** for
`tag:conductor-yamkit` only. Generate the credential and copy its client secret.
The bootstrap needs only the secret, not the client ID.

## 3. Configure Conductor Cloud once

First put this change on the branch new workspaces use, normally `main`. The
setup command needs `scripts/cloud_tailscale.py` and `scripts/tailscale` in the
new workspace checkout.

In **Settings → Organization → Cloud Computer**:

1. Add `YAMKIT_TAILSCALE_OAUTH_SECRET` under **Environment**, with the OAuth
   client secret as its value. This environment is organization-wide. Save the
   value there; do not put it in git, rig.yaml, chat, or a setup-script literal.
2. Under the **yamkit repository's Setup script**, append:

   ```bash
   python3 scripts/cloud_tailscale.py setup
   ```

   Preserve any existing project setup commands. This bootstrap only prepares
   private networking; it does not install yamkit's Python dependencies.
3. Choose **Build computer** so new workspaces inherit the secret. Existing
   workspaces retain their original environment. Saving only a setup-script
   change does not require a build.

Current Conductor Cloud runs the repository setup configured on the Cloud
Computer; it does not run `scripts.setup` from `.conductor/settings.toml` or
`conductor.json`. Regular workspace tools cannot edit the persistent cloud
environment. **Configure with an agent** can help with the setup script and
build, but saving the secret is a user action.
See [Cloud Computer settings](https://www.conductor.build/docs/cloud/cloud-computer)
and [cloud environment variables](https://www.conductor.build/docs/cloud/environment-variables).

## Use from every new workspace

```bash
scripts/tailscale status
scripts/tailscale ssh andre@yam-lenovo 'hostname; whoami'
```

`scripts/tailscale` starts or reconnects the workspace client before forwarding
the command. This also handles cloud sleep: Conductor preserves files but stops
processes, and creation-time setup is not a guarantee of setup on wake.
See [workspace lifecycle](https://www.conductor.build/docs/cloud/working-with-cloud-workspaces).

The current Lenovo project is `/home/andre/rohan-new`. Inspect its README,
instructions, and working-tree status before editing it. Multiple cloud
workspaces share that physical machine and rig.

The client uses userspace networking. Use `scripts/tailscale ssh` for SSH and
port forwarding; plain system SSH has no Tailscale network interface here.
All binaries, sockets, logs, temporary files, and SSH host keys stay under this
checkout's ignored `.tools/tailscale/` directory. The bootstrap downloads pinned
official static binaries and verifies their checksums. The OAuth secret is
passed through a private temporary file during enrollment, then removed.

## Troubleshooting

- **Missing OAuth secret:** save `YAMKIT_TAILSCALE_OAUTH_SECRET` in Cloud Computer
  Environment, build the computer, then create a workspace from that build.
- **Enrollment rejected:** check the credential's Auth Keys Write scope and its
  permission for `tag:conductor-yamkit`.
- **SSH denied:** verify Lenovo has `tag:yam-lenovo`, Tailscale SSH is enabled,
  and both the network grant and SSH accept rule above are present.
- **Lenovo offline:** power/network or Lenovo node-key expiry needs attention.
  Workspace enrollment does not power on the rig computer.
- **After sleep:** run the same `scripts/tailscale` command; it starts a new
  client identity automatically using the saved credential.

Connection setup never starts teleop, recording, rollout, calibration, or any
other motor command. Existing speed clamps and firmware timeouts stay in place.
