let accessToken;
const clientId = 'cbd9d00eaaad4865b8f666bf3d994054';
const redirectUri = 'http://jobless-pop.surge.sh';

// npm install --global surge
// redirectUri 
// npm run build
// npx surge

const Spotify = {
    getAccessToken() {
        if (accessToken) {
            return accessToken;
        }
        // check for access token
        const accessTokenMatch = window.location.href.match(/access_token=([^&]*)/);
        // capture when the token expires
        const expiresInMatch = window.location.href.match(/expires_in=([^&]*)/);

        if (accessTokenMatch && expiresInMatch) {
            accessToken = accessTokenMatch[1];
            const expiresIn = Number(expiresInMatch[1])
            // This clears the parameters, allowing us to grab a new access token when it expires
            window.setTimeout( ()=> accessToken = '', expiresIn * 1000);
            window.history.pushState('Access Token', null, '/');
            return accessToken;

        }

        else {
            const accessUrl = `https://accounts.spotify.com/authorize?client_id=${clientId}&response_type=token&scope=playlist-modify-public&redirect_uri=${redirectUri}`;
            window.location = accessUrl;

        }
    },

    search(term) {
        const access_token = Spotify.getAccessToken();
        return fetch(`https://api.spotify.com/v1/search?type=track&q=${term}`, {
            headers: {Authorization: `Bearer ${access_token}`}
          }).then(response => {
            return response.json();
          }).then(jsonResponse => {
            if (!jsonResponse.tracks) {
                return [];
            }
            return jsonResponse.tracks.items.map(track => ({
                id: track.id,
                name: track.name,
                artists: track.artists[0].name,
                album: track.album.name,
                uri: track.uri
            }));
          })
    },

    savePlaylist(name, trackUris) {
        if (!name || !trackUris.length) { 
            return ;
        }

        const access_token = Spotify.getAccessToken();
        const headers = {Authorization: `Bearer ${access_token}`};
        let user_id;
        
        return fetch('https://api.spotify.com/v1/me', {headers: headers}).then(
            response => response.json()
        ).then(jsonResponse => {
            user_id = jsonResponse.id;
            return fetch(`https://api.spotify.com/v1/users/${user_id}/playlists`, {
                headers: headers, 
                method: 'POST',
                body: JSON.stringify({name: name})
            }).then(response => response.json()).then(jsonResponse => {
                const playlistID = jsonResponse.id;
                return fetch(`https://api.spotify.com/v1/users/${user_id}/playlists/${playlistID}/tracks`, {
                    headers: headers,
                    method: 'POST',
                    body: JSON.stringify({uris: trackUris})
                })
            })
        })
    }





    }

export default Spotify;