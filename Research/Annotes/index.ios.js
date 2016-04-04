/**
 * Sample React Native App
 * https://github.com/facebook/react-native
 */

import React, {
  AppRegistry,
  Component,
  StyleSheet,
  Text,
  View,
  Dimensions,
  NavigatorIOS,
  TabBarIOS,
} from 'react-native';

var GiftedMessenger = require('react-native-gifted-messenger');
var ScrollableTabView=require('react-native-scrollable-tab-view');
var Icon=require('react-native-vector-icons/FontAwesome');

var GiftedMessengerExample = React.createClass({
  getMessages() {
    return [
      {text: 'Are you building a chat app?', name: 'React-Native', image: {uri: 'https://facebook.github.io/react/img/logo_og.png'}, position: 'left', date: new Date(2015, 0, 16, 19, 0)},
      {text: "Yes, and I use Gifted Messenger!", name: 'Developer', image: null, position: 'right', date: new Date(2015, 0, 17, 19, 0)},
    ];
  },
  handleSend(message = {}, rowID = null) {
    // Send message.text to your server
  },
  handleReceive() {
    this._GiftedMessenger.appendMessage({
      text: 'Received message',
      name: 'Friend',
      image: {uri: 'https://facebook.github.io/react/img/logo_og.png'},
      position: 'left',
      date: new Date(),
    });
  },
  render() {
    return (
      <GiftedMessenger
        ref={(c) => this._GiftedMessenger = c}

        messages={this.getMessages()}
        handleSend={this.handleSend}
        maxHeight={Dimensions.get('window').height - 64} // 64 for the navBar
        placeholder={"メッセージを入力してください"}
        sendButtonText={"送信"}
        autoFocus={false}
        styles={{
          container:{
            backgroundColor:'#FFF0FF'
          },
          bubbleLeft: {
            backgroundColor: '#e6e6eb',
            marginRight: 70,
          },
          bubbleRight: {
            backgroundColor: '#007aff',
            marginLeft: 70,
          },
        }}
      />
    );
  },
});

class SamplePage extends Component {
  render() {
    return (
      <View style={styles.container}>
        <Text style={styles.welcome}>
          Welcome to React Native!
        </Text>
        <Text style={styles.instructions}>
          To get started, edit index.ios.js
        </Text>
        <Text style={styles.instructions}>
          Press Cmd+R to reload,{'\n'}
          Cmd+D or shake for dev menu
        </Text>
      </View>
    );
  }
}

var GrilleMain=React.createClass({
  getInitialState(){
    return{
      selectedTab:"default"
    };
  },
  _renderContent(category:string,title:?string){
    return(
      <NavigatorIOS
        style={style.containerp}
        initialRoute={{
          component:GiftedMessengerExample,
          title:title,
          passProps:{filter:category}
        }}
      />
    );
  },
  render(){
    return(
      <Icon.TabBarIOS>
      <Icon.TabBarItem
        title="All"
        iconName="dribbble"
        selectedIconName="dribbble"
        selected={this.state.selectedTab === "default"}
        onPress={() => {
          this.setState({
            selectedTab: "default",
          });
        }}>
        {this._renderContent("default", "All")}
      </Icon.TabBarItem>
      </Icon.TabBarIOS>
    );
  }
});


class Annotes extends Component {
  render() {
    return (
      <NavigatorIOS
        style={styles.containerp}
        tintColor='#FF6600'
        initialRoute={{
          title:'Annotes',
          component:GiftedMessengerExample,
        }}/>
    );
  }
}

const styles = StyleSheet.create({
  containerp:{
    flex:1,
    backgroundColor:'#F6F6EF',
  },
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#F5FCFF',
  },
  welcome: {
    fontSize: 20,
    textAlign: 'center',
    margin: 10,
  },
  instructions: {
    textAlign: 'center',
    color: '#333333',
    marginBottom: 5,
  },
});

AppRegistry.registerComponent('Annotes', () => Annotes);
